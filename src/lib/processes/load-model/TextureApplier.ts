import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { GLTFExporter } from 'three/examples/jsm/exporters/GLTFExporter.js'
import * as THREE from 'three'

/**
 * Takes a GLB model and a 2D image, and applies that image as the
 * model's texture using a simple front-projection (planar) UV mapping,
 * generated fresh for this purpose - it deliberately IGNORES whatever
 * UVs the model already has.
 *
 * Why: a model's existing UVs (e.g. from an AI texture-synthesis atlas,
 * or from xatlas unwrapping for painting) are laid out as scattered
 * "charts" optimized for packing efficiency, not for spatial
 * correspondence to a front-facing photo. Wrapping a photo onto that
 * kind of UV layout produces exactly the "random confetti of image
 * fragments with big blank gaps" result this was built to avoid -
 * confirmed by testing, not hypothetical.
 *
 * Front-projection instead maps each vertex's U to its X position and V
 * to its Y position (both normalized to the model's bounding box, front
 * view) - so the image visually corresponds to the model the way you'd
 * expect a front-facing photo applied to a standing character to look:
 * left-right and up-down match, the image reads correctly from the
 * front. The back of the model gets a mirrored copy of the same
 * projection (there's no information about the back in a single photo,
 * so this is the most reasonable default) rather than a
 * disconnected fragment of unrelated UV atlas.
 *
 * This is NOT multi-view AI texture synthesis (like Hugging Face's
 * mv-adapter) - there's no intelligence about which image region
 * belongs on which body part beyond simple front-view geometry. It's
 * the more sensible of the two non-AI options, not a replacement for
 * real texture synthesis.
 */
export class TextureApplier {
  private readonly gltf_loader = new GLTFLoader()
  private readonly gltf_exporter = new GLTFExporter()

  private on_progress: (message: string) => void = () => {}

  public set_progress_callback (callback: (message: string) => void): void {
    this.on_progress = callback
  }

  /**
   * Resolves with a blob: URL for a new GLB with the given image applied
   * as the texture map on every mesh, using fresh front-projection UVs.
   */
  public async apply_texture (glb_url: string, image_file: File): Promise<string> {
    this.on_progress('Loading model…')
    const gltf = await this.gltf_loader.loadAsync(glb_url)
    const scene = gltf.scene

    const meshes: THREE.Mesh[] = []
    scene.traverse((child) => {
      if (child instanceof THREE.Mesh) {
        meshes.push(child)
      }
    })
    this.on_progress(`DEBUG: model loaded, ${meshes.length} mesh(es) found`)

    if (meshes.length === 0) {
      throw new Error('No mesh found in this model.')
    }

    this.on_progress('Loading image…')
    const image_object_url = URL.createObjectURL(image_file)
    const texture = await new THREE.TextureLoader().loadAsync(image_object_url)
    // NOTE: intentionally NOT revoking image_object_url here - it stays
    // alive until this whole method returns, in case the exporter needs
    // to re-read the image data during embedding

    const image_element = texture.image as HTMLImageElement | undefined
    this.on_progress(
      `DEBUG: image loaded, dimensions=${image_element?.naturalWidth ?? '?'}x${image_element?.naturalHeight ?? '?'}`
    )

    texture.colorSpace = THREE.SRGBColorSpace
    texture.needsUpdate = true

    // compute ONE shared bounding box across all meshes, so multi-mesh
    // models get a single consistent projection rather than each mesh
    // being independently stretched to fill the image
    const combined_bounds = new THREE.Box3().setFromObject(scene)

    this.on_progress(`Applying front-projected texture to ${meshes.length} mesh(es)…`)
    for (const mesh of meshes) {
      this.generate_front_projection_uvs(mesh, combined_bounds)
      const new_material = new THREE.MeshStandardMaterial({ map: texture })
      mesh.material = new_material
    }

    this.on_progress('Exporting textured model…')
    let exported_buffer: ArrayBuffer
    try {
      exported_buffer = await new Promise<ArrayBuffer>((resolve, reject) => {
        this.gltf_exporter.parse(
          scene,
          (result) => {
            if (result instanceof ArrayBuffer) {
              resolve(result)
            } else {
              reject(new Error('Textured export did not return binary GLB data'))
            }
          },
          (error: unknown) => {
            reject(error instanceof Error ? error : new Error(String(error)))
          },
          { binary: true, onlyVisible: false, embedImages: true }
        )
      })
    } finally {
      URL.revokeObjectURL(image_object_url)
    }

    this.on_progress(`DEBUG: export complete, ${exported_buffer.byteLength} bytes`)

    return URL.createObjectURL(new Blob([exported_buffer], { type: 'model/gltf-binary' }))
  }

  /**
   * Overwrites a mesh's UVs with a simple front-view planar projection:
   * U from world-space X, V from world-space Y, both normalized against
   * the shared bounding box passed in. Deliberately replaces any
   * existing UVs - see class-level comment for why.
   */
  private generate_front_projection_uvs (mesh: THREE.Mesh, bounds: THREE.Box3): void {
    const geometry = mesh.geometry
    const position_attribute = geometry.attributes.position
    if (position_attribute === undefined) {
      return
    }

    const size = new THREE.Vector3()
    bounds.getSize(size)
    const width = Math.max(size.x, 1e-6)
    const height = Math.max(size.y, 1e-6)

    const vertex_count = position_attribute.count
    const uv_array = new Float32Array(vertex_count * 2)
    const world_position = new THREE.Vector3()

    mesh.updateWorldMatrix(true, false)

    for (let i = 0; i < vertex_count; i++) {
      world_position.set(
        position_attribute.getX(i),
        position_attribute.getY(i),
        position_attribute.getZ(i)
      )
      world_position.applyMatrix4(mesh.matrixWorld)

      uv_array[i * 2] = (world_position.x - bounds.min.x) / width
      uv_array[i * 2 + 1] = (world_position.y - bounds.min.y) / height
    }

    geometry.setAttribute('uv', new THREE.BufferAttribute(uv_array, 2))
  }
}
