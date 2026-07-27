import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { GLTFExporter } from 'three/examples/jsm/exporters/GLTFExporter.js'
import { Mesh, BufferAttribute, type BufferGeometry } from 'three'
import SimplifyWorkerConstructor from './SimplifyWorker.ts?worker'
import type { SimplifyRequest, SimplifyResponse } from './SimplifyWorker.ts'

/**
 * Reduces the polygon count of a GLB model entirely client-side.
 *
 * Runs meshoptimizer inside a Web Worker (see SimplifyWorker.ts) rather
 * than on the main thread - confirmed in practice that simplifying a very
 * dense (~1M triangle) AI-generated mesh directly on the main thread can
 * freeze the page for many seconds with no feedback, even though
 * meshoptimizer itself is fast relative to alternatives.
 *
 * Intended to run on freshly-generated (image-to-3D) meshes BEFORE the
 * skeleton-fitting step, since simplification does not understand skin
 * weights - it should only ever touch un-rigged geometry.
 */
export class MeshFaceReducer {
  private readonly gltf_loader = new GLTFLoader()
  private readonly gltf_exporter = new GLTFExporter()

  private on_progress: (message: string) => void = () => {}

  public set_progress_callback (callback: (message: string) => void): void {
    this.on_progress = callback
  }

  /**
   * Loads the GLB at glb_url, reduces every mesh in it toward
   * target_face_count (distributed proportionally across meshes if there
   * are more than one), re-exports as a new GLB, and resolves with a
   * fresh blob: URL. The original glb_url is left untouched.
   */
  public async reduce_glb_face_count (glb_url: string, target_face_count: number): Promise<string> {
    const gltf = await this.gltf_loader.loadAsync(glb_url)
    const scene = gltf.scene

    const meshes: Mesh[] = []
    scene.traverse((child) => {
      if (child instanceof Mesh) {
        meshes.push(child)
      }
    })

    const total_faces_before = meshes.reduce((sum, mesh) => sum + this.count_faces(mesh.geometry), 0)

    if (total_faces_before === 0 || target_face_count >= total_faces_before) {
      // nothing to reduce, or target is not actually smaller
      return glb_url
    }

    const reduction_ratio = target_face_count / total_faces_before

    for (let i = 0; i < meshes.length; i++) {
      this.on_progress(`Reducing mesh ${i + 1} of ${meshes.length}…`)
      await this.simplify_mesh(meshes[i], reduction_ratio)
    }

    this.on_progress('Exporting reduced model…')
    const exported_buffer = await new Promise<ArrayBuffer>((resolve, reject) => {
      this.gltf_exporter.parse(
        scene,
        (result) => {
          if (result instanceof ArrayBuffer) {
            resolve(result)
          } else {
            reject(new Error('Face-reduced export did not return binary GLB data'))
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

  private async simplify_mesh (mesh: Mesh, reduction_ratio: number): Promise<void> {
    const geometry = mesh.geometry
    const position_attribute = geometry.attributes.position
    if (position_attribute === undefined) {
      return
    }

    const current_faces = this.count_faces(geometry)
    if (current_faces <= 4) {
      return // too small to usefully simplify
    }

    const target_index_count = Math.max(12, Math.round(current_faces * reduction_ratio) * 3)

    const existing_index = geometry.getIndex()
    const indices = existing_index !== null
      ? Uint32Array.from(existing_index.array)
      : Uint32Array.from({ length: position_attribute.count }, (_, i) => i) // non-indexed: 0,1,2,3...

    const positions = position_attribute.array instanceof Float32Array
      ? position_attribute.array.slice()
      : Float32Array.from(position_attribute.array)

    const request: SimplifyRequest = { positions, indices, target_index_count }
    const response = await this.run_simplify_worker(request, 60_000)

    if (response.status === 'error') {
      throw new Error(response.message)
    }

    if (response.indices.length < 3) {
      return // simplification failed to produce a usable mesh, leave as-is
    }

    geometry.setIndex(new BufferAttribute(response.indices, 1))
  }

  private async run_simplify_worker (request: SimplifyRequest, timeout_ms: number): Promise<SimplifyResponse> {
    const worker = new SimplifyWorkerConstructor()

    try {
      return await new Promise<SimplifyResponse>((resolve, reject) => {
        const timeout_handle = window.setTimeout(() => {
          reject(new Error(`Face reduction timed out after ${Math.round(timeout_ms / 1000)}s.`))
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

  private count_faces (geometry: BufferGeometry): number {
    const index = geometry.getIndex()
    const vertex_count = index !== null ? index.count : geometry.attributes.position.count
    return Math.floor(vertex_count / 3)
  }
}
