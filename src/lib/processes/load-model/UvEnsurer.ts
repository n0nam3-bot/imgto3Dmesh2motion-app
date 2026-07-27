import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { GLTFExporter } from 'three/examples/jsm/exporters/GLTFExporter.js'
import * as THREE from 'three'
import UvUnwrapWorkerConstructor from './UvUnwrapWorker.ts?worker'
import type { UvUnwrapRequest, UvUnwrapResponse } from './UvUnwrapWorker.ts'

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
      this.on_progress(`Generating UVs for mesh ${i + 1} of ${meshes_missing_uvs.length}…`)
      await this.generate_real_uvs(meshes_missing_uvs[i].geometry)
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
    const response = await this.run_uv_unwrap_worker(request, 60_000)

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
