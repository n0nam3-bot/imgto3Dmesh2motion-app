import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { GLTFExporter } from 'three/examples/jsm/exporters/GLTFExporter.js'
import * as THREE from 'three'
import { UvEnsurer } from './UvEnsurer.ts'

/**
 * Takes a GLB model and a 2D image, and applies that image directly as
 * the diffuse texture map across the model's UV layout. Not a multi-view
 * texture projection (that would require the exact camera used to
 * generate the model, which isn't available) - this is the simpler,
 * direct approach: the image IS the texture, wrapped according to
 * whatever UVs the model has (real ones if present, auto-generated via
 * xatlas otherwise).
 */
export class TextureApplier {
  private readonly gltf_loader = new GLTFLoader()
  private readonly gltf_exporter = new GLTFExporter()
  private readonly uv_ensurer = new UvEnsurer()

  private on_progress: (message: string) => void = () => {}

  public set_progress_callback (callback: (message: string) => void): void {
    this.on_progress = callback
    this.uv_ensurer.set_progress_callback(callback)
  }

  /**
   * Resolves with a blob: URL for a new GLB with the given image applied
   * as the texture map on every mesh.
   */
  public async apply_texture (glb_url: string, image_file: File): Promise<string> {
    this.on_progress('DEBUG: starting apply_texture')
    this.on_progress('Checking model UV coordinates…')
    const uv_ready_url = await this.uv_ensurer.ensure_uvs(glb_url)
    this.on_progress('DEBUG: UV check done')

    this.on_progress('Loading model…')
    const gltf = await this.gltf_loader.loadAsync(uv_ready_url)
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
      `DEBUG: image loaded, dimensions=${image_element?.naturalWidth ?? '?'}x${image_element?.naturalHeight ?? '?'}, ` +
      `complete=${image_element?.complete ?? '?'}`
    )

    texture.colorSpace = THREE.SRGBColorSpace
    texture.needsUpdate = true

    this.on_progress(`Applying texture to ${meshes.length} mesh(es)…`)
    for (const mesh of meshes) {
      const has_uv = mesh.geometry.attributes.uv !== undefined
      this.on_progress(`DEBUG: mesh "${mesh.name || '(unnamed)'}" has_uv=${has_uv}`)
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
}
