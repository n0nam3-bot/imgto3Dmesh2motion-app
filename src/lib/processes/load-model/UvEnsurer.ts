import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { GLTFExporter } from 'three/examples/jsm/exporters/GLTFExporter.js'
import * as THREE from 'three'
import UvUnwrapWorkerConstructor from './UvUnwrapWorker.ts?worker'
import type { UvUnwrapRequest, UvUnwrapResponse } from './UvUnwrapWorker.ts'
import SimplifyWorkerConstructor from './SimplifyWorker.ts?worker'
import type { SimplifyRequest, SimplifyResponse } from './SimplifyWorker.ts'

// xatlas unwrapping time grows steeply with triangle count - meshes denser
// than this get pre-reduced (for the purpose of unwrapping ONLY) so the
// unwrap step stays reliably fast, regardless of the user's own face-count
// preference (which is applied separately, before this step ever runs)
const MAX_FACES_BEFORE_UNWRAP = 40_000

/**
 * Automatically checks a GLB for UV coordinates and generates them (via
 * xatlas, run in a worker) for any mesh missing them, so downstream steps
 * (texturing, export, external tools) always have something to work with.
 * If every mesh already has UVs, the input is returned unchanged - no
 * re-export, no wasted work.
 */
export class UvEnsurer {
  private readonly gltf_loader = new GLTFLoader()
  private readonly gltf_exporter = new GLTFExporter()

  private on_progress: (message: string) => void = () => {}

  public set_progress_callback (callback: (message: string) => void): void {
    this.on_progress = callback
  }

  /**
   * Loads the GLB at glb_url. If every mesh already has UVs, resolves
   * with the SAME url unchanged. Otherwise generates UVs for whichever
   * meshes are missing them, re-exports, and resolves with a fresh
   * blob: URL.
   */
  public async ensure_uvs (glb_url: string): Promise<string> {
    const gltf = await this.gltf_loader.loadAsync(glb_url)
    const scene = gltf.scene

    const meshes: THREE.Mesh[] = []
    scene.traverse((child) => {
      if (child instanceof THREE.Mesh) {
        meshes.push(child)
      }
    })

    const meshes_missing_uvs = meshes.filter((mesh) => mesh.geometry.attributes.uv === undefined)

    if (meshes_missing_uvs.length === 0) {
      return glb_url // nothing to do
    }

    for (let i = 0; i < meshes_missing_uvs.length; i++) {
      const mesh = meshes_missing_uvs[i]
      const face_count = this.count_faces(mesh.geometry)

      if (face_count > MAX_FACES_BEFORE_UNWRAP) {
        this.on_progress(
          `Mesh ${i + 1} of ${meshes_missing_uvs.length} is very dense (${face_count} faces) - ` +
          `reducing to ~${MAX_FACES_BEFORE_UNWRAP} faces so UV unwrapping stays fast…`
        )
        await this.simplify_for_unwrap(mesh, MAX_FACES_BEFORE_UNWRAP)
      }

      this.on_progress(`Generating UVs for mesh ${i + 1} of ${meshes_missing_uvs.length}…`)
      await this.generate_real_uvs(mesh.geometry)
    }

    this.on_progress('Exporting model with generated UVs…')
    const exported_buffer = await new Promise<ArrayBuffer>((resolve, reject) => {
      this.gltf_exporter.parse(
        scene,
        (result) => {
          if (result instanceof ArrayBuffer) {
            resolve(result)
          } else {
            reject(new Error('UV-fixed export did not return binary GLB data'))
          }
        },
        (error: unknown) => {
          reject(error instanceof Error ? error : new Error(String(error)))
        },
        { binary: true, onlyVisible: false, embedImages: true }
      )
    })

    return URL.createObjectURL(new Blob([exported_buffer], { type: 'model/gltf-binary' }))
  }

  private count_faces (geometry: THREE.BufferGeometry): number {
    const index = geometry.getIndex()
    const vertex_count = index !== null ? index.count : geometry.attributes.position.count
    return Math.floor(vertex_count / 3)
  }

  /**
   * Reduces a mesh toward target_face_count using meshoptimizer (in a
   * worker), purely to keep the UV unwrap step tractable on very dense
   * meshes. This is separate from - and unrelated to - the user's own
   * face-count preference from MeshFaceReducer.
   */
  private async simplify_for_unwrap (mesh: THREE.Mesh, target_face_count: number): Promise<void> {
    const geometry = mesh.geometry
    const position_attribute = geometry.attributes.position
    const current_faces = this.count_faces(geometry)
    if (position_attribute === undefined || current_faces <= target_face_count) {
      return
    }

    const target_index_count = Math.max(12, target_face_count * 3)

    const existing_index = geometry.getIndex()
    const indices = existing_index !== null
      ? Uint32Array.from(existing_index.array)
      : Uint32Array.from({ length: position_attribute.count }, (_, i) => i)

    const positions = position_attribute.array instanceof Float32Array
      ? position_attribute.array.slice()
      : Float32Array.from(position_attribute.array)

    const request: SimplifyRequest = { positions, indices, target_index_count }
    const response = await this.run_simplify_worker(request, 60_000)

    if (response.status === 'error') {
      throw new Error(response.message)
    }
    if (response.indices.length < 3) {
      return
    }

    geometry.setIndex(new THREE.BufferAttribute(response.indices, 1))
    // topology changed significantly - stale normals would cause the
    // same "gray/blotchy" look fixed in MeshFaceReducer
    geometry.computeVertexNormals()
  }

  private async run_simplify_worker (request: SimplifyRequest, timeout_ms: number): Promise<SimplifyResponse> {
    const worker = new SimplifyWorkerConstructor()

    try {
      return await new Promise<SimplifyResponse>((resolve, reject) => {
        const timeout_handle = window.setTimeout(() => {
          reject(new Error(`Pre-unwrap simplification timed out after ${Math.round(timeout_ms / 1000)}s.`))
        }, timeout_ms)

        worker.onmessage = (event: MessageEvent<SimplifyResponse>) => {
          window.clearTimeout(timeout_handle)
          resolve(event.data)
        }

        worker.onerror = (error: ErrorEvent) => {
          window.clearTimeout(timeout_handle)
          reject(new Error(error.message))
        }

        const transfer_list: ArrayBuffer[] = [request.positions.buffer, request.indices.buffer]
        worker.postMessage(request, transfer_list)
      })
    } finally {
      worker.terminate()
    }
  }

  /**
   * Generates real UV coordinates for a mesh that has none, using xatlas
   * (same UV-unwrapping library used by Blender's own glTF exporter),
   * run in a Web Worker so it can't freeze the page regardless of mesh
   * density, and so a timeout can actually terminate it if needed.
   */
  private async generate_real_uvs (geometry: THREE.BufferGeometry): Promise<void> {
    const position_attribute = geometry.attributes.position
    if (position_attribute === undefined) {
      return
    }

    const positions = position_attribute.array instanceof Float32Array
      ? position_attribute.array.slice()
      : Float32Array.from(position_attribute.array)

    const index_attribute = geometry.getIndex()
    const indices = index_attribute !== null
      ? (index_attribute.array instanceof Uint32Array
          ? index_attribute.array.slice()
          : Uint32Array.from(index_attribute.array))
      : undefined

    const normal_attribute = geometry.attributes.normal
    const normals = normal_attribute !== undefined
      ? (normal_attribute.array instanceof Float32Array
          ? normal_attribute.array.slice()
          : Float32Array.from(normal_attribute.array))
      : undefined

    const request: UvUnwrapRequest = { positions, indices, normals }
    // 120s - generous since a worker running long doesn't freeze the page,
    // and the MAX_FACES_BEFORE_UNWRAP cap above should keep this well
    // within that in practice
    const response = await this.run_uv_unwrap_worker(request, 120_000)

    if (response.status === 'error') {
      throw new Error(response.message)
    }
    if (response.status === 'progress') {
      throw new Error('Unexpected progress message reached response handling')
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(response.positions, 3))
    geometry.setAttribute('uv', new THREE.BufferAttribute(response.uvs, 2))
    if (response.normals !== undefined) {
      geometry.setAttribute('normal', new THREE.BufferAttribute(response.normals, 3))
    } else {
      geometry.computeVertexNormals()
    }
    geometry.setIndex(new THREE.BufferAttribute(response.indices, 1))
    geometry.computeBoundingSphere()
    geometry.computeBoundingBox()
  }

  private async run_uv_unwrap_worker (request: UvUnwrapRequest, timeout_ms: number): Promise<UvUnwrapResponse> {
    const worker = new UvUnwrapWorkerConstructor()

    try {
      return await new Promise<UvUnwrapResponse>((resolve, reject) => {
        const timeout_handle = window.setTimeout(() => {
          reject(new Error(`UV unwrapping timed out after ${Math.round(timeout_ms / 1000)}s.`))
        }, timeout_ms)

        worker.onmessage = (event: MessageEvent<UvUnwrapResponse>) => {
          if (event.data.status === 'progress') {
            this.on_progress(event.data.message)
            return
          }
          window.clearTimeout(timeout_handle)
          resolve(event.data)
        }

        worker.onerror = (error: ErrorEvent) => {
          window.clearTimeout(timeout_handle)
          reject(new Error(error.message))
        }

        const transfer_list: ArrayBuffer[] = [request.positions.buffer]
        if (request.indices !== undefined) {
          transfer_list.push(request.indices.buffer)
        }
        if (request.normals !== undefined) {
          transfer_list.push(request.normals.buffer)
        }

        worker.postMessage(request, transfer_list)
      })
    } finally {
      worker.terminate()
    }
  }
}
