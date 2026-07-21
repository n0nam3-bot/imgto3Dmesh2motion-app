import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { GLTFExporter } from 'three/examples/jsm/exporters/GLTFExporter.js'
import { SimplifyModifier } from 'three/examples/jsm/modifiers/SimplifyModifier.js'
import { Mesh, type BufferGeometry } from 'three'

/**
 * Reduces the polygon count of a GLB model entirely client-side, using
 * three.js's own SimplifyModifier. Intended to run on freshly-generated
 * (image-to-3D) meshes BEFORE the skeleton-fitting step, since the
 * modifier does not understand skin weights - it should only ever touch
 * un-rigged geometry.
 */
export class MeshFaceReducer {
  private readonly gltf_loader = new GLTFLoader()
  private readonly gltf_exporter = new GLTFExporter()
  private readonly simplify_modifier = new SimplifyModifier()

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

    for (const mesh of meshes) {
      const current_faces = this.count_faces(mesh.geometry)
      if (current_faces <= 4) {
        continue // too small to usefully simplify
      }

      const mesh_target_faces = Math.max(4, Math.round(current_faces * reduction_ratio))
      const current_vertices = mesh.geometry.attributes.position.count
      // faces:vertices is roughly 2:1 for a closed triangle mesh (Euler's formula),
      // so approximate the target vertex count the same way
      const target_vertices = Math.max(4, Math.round(mesh_target_faces / 2))
      const vertices_to_remove = Math.max(0, current_vertices - target_vertices)

      if (vertices_to_remove > 0) {
        mesh.geometry = this.simplify_modifier.modify(mesh.geometry, vertices_to_remove)
      }
    }

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

  private count_faces (geometry: BufferGeometry): number {
    const index = geometry.getIndex()
    const vertex_count = index !== null ? index.count : geometry.attributes.position.count
    return Math.floor(vertex_count / 3)
  }
}
