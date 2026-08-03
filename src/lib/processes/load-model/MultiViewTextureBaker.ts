import * as THREE from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { GLTFExporter } from 'three/examples/jsm/exporters/GLTFExporter.js'

/**
 * Bakes a set of multi-view images (each taken from a known camera angle
 * around the mesh) directly onto the mesh's existing UV layout, entirely
 * client-side - no server-side texturing step required.
 *
 * WHY THIS EXISTS: VAST-AI/MV-Adapter-Img2Texture's own /run_texturing
 * step (which does this same job server-side) was confirmed broken -
 * failing with the same ZeroGPU worker ValueError even through the
 * Space's own web UI, not just our API calls. Its /run_mvadapter step
 * (generating the 6 view images) works reliably. This class replaces
 * only the broken half of that pipeline.
 *
 * TECHNIQUE (standard multi-view texture baking):
 * For each of the 6 known camera angles, the mesh is rendered "unfolded"
 * into UV space - i.e. each vertex's clip-space position is set directly
 * from its UV coordinate rather than from camera projection. Within that
 * UV-space rasterization, each fragment separately computes where that
 * same point WOULD project to in the corresponding view image (using
 * that camera's real view/projection matrices), and samples color from
 * there. A depth-buffer trick (writing 1 - facing_quality as depth, with
 * standard depth testing) means that across all 6 passes rendered into
 * the same target, each UV texel keeps only the sample from whichever
 * camera saw that surface most directly - a standard technique for
 * combining several viewpoints without cross-channel color artifacts.
 *
 * MAIN CALIBRATION RISK (likely needs iteration): the exact camera
 * distance/FOV/elevation used to angles 0-5 aren't published by the
 * Space - only the azimuth angles were recoverable from their source
 * (camera_azimuth_deg=[x-90 for x in [0,90,180,270,180,180]] ->
 * [-90,0,90,180,90,90]). The repeated 90 for views 4-5 strongly suggests
 * those are elevation-varied (top/bottom) rather than distinct azimuths;
 * DEFAULT_VIEW_ANGLES below is a reasonable guess at that split, not a
 * confirmed value - if results look misaligned, this is the first thing
 * to adjust.
 */

export interface ViewAngle {
  azimuth_deg: number
  elevation_deg: number
}

// see class-level comment - views 0-3 (the four distinct azimuths) are
// confirmed by source; views 4-5 (top/bottom) are an educated guess
export const DEFAULT_VIEW_ANGLES: ViewAngle[] = [
  { azimuth_deg: -90, elevation_deg: 0 }, // left
  { azimuth_deg: 0, elevation_deg: 0 }, // front
  { azimuth_deg: 90, elevation_deg: 0 }, // right
  { azimuth_deg: 180, elevation_deg: 0 }, // back
  { azimuth_deg: 45, elevation_deg: 65 }, // top (guessed)
  { azimuth_deg: 45, elevation_deg: -65 } // bottom (guessed)
]

const UNFOLD_VERTEX_SHADER = `
precision highp float;
uniform mat4 cameraViewMatrix;
uniform mat4 cameraProjectionMatrix;
uniform vec3 cameraWorldPosition;

out vec2 vScreenUV;
out float vFacing;

void main() {
  vec4 worldPos = modelMatrix * vec4(position, 1.0);
  vec3 worldNormal = normalize(mat3(modelMatrix) * normal);

  vec4 camClip = cameraProjectionMatrix * cameraViewMatrix * worldPos;
  vScreenUV = (camClip.xy / camClip.w) * 0.5 + 0.5;

  vec3 toCam = normalize(cameraWorldPosition - worldPos.xyz);
  vFacing = dot(worldNormal, toCam);

  // rasterize in UV space, not camera space - this "unfolds" the mesh
  // onto the output texture regardless of where the sample camera is
  gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}
`

const UNFOLD_FRAGMENT_SHADER = `
precision highp float;
uniform sampler2D viewImage;
in vec2 vScreenUV;
in float vFacing;
out vec4 outColor;

void main() {
  if (vFacing <= 0.05) discard;
  if (vScreenUV.x < 0.0 || vScreenUV.x > 1.0 || vScreenUV.y < 0.0 || vScreenUV.y > 1.0) discard;

  outColor = texture(viewImage, vScreenUV);
  // standard (not inverted) depth test below means SMALLER depth wins -
  // so encode higher facing quality as smaller depth
  gl_FragDepth = 1.0 - clamp(vFacing, 0.0, 1.0);
}
`

export class MultiViewTextureBaker {
  private readonly gltf_loader = new GLTFLoader()
  private readonly gltf_exporter = new GLTFExporter()
  private readonly texture_loader = new THREE.TextureLoader()
  private readonly renderer = new THREE.WebGLRenderer({ antialias: false })

  private on_progress: (message: string) => void = () => {}

  public set_progress_callback (callback: (message: string) => void): void {
    this.on_progress = callback
  }

  /**
   * Bakes the given view images (in the same order as view_angles) onto
   * the mesh at glb_url, and resolves with a blob: URL for the result.
   * The mesh MUST already have UV coordinates (this app's pipeline
   * ensures that earlier via UvEnsurer before this point is ever
   * reached).
   */
  public async bake (
    glb_url: string,
    view_image_urls: string[],
    view_angles: ViewAngle[] = DEFAULT_VIEW_ANGLES,
    output_resolution: number = 1024
  ): Promise<string> {
    if (view_image_urls.length !== view_angles.length) {
      throw new Error(
        `Got ${view_image_urls.length} view image(s) but ${view_angles.length} view angle(s) - these must match.`
      )
    }

    this.on_progress('Loading mesh…')
    const gltf = await this.gltf_loader.loadAsync(glb_url)
    const scene = gltf.scene

    const meshes: THREE.Mesh[] = []
    scene.traverse((child) => {
      if (child instanceof THREE.Mesh) {
        meshes.push(child)
      }
    })
    if (meshes.length === 0) {
      throw new Error('No mesh found in this model.')
    }
    const mesh_missing_uv = meshes.find((mesh) => mesh.geometry.attributes.uv === undefined)
    if (mesh_missing_uv !== undefined) {
      throw new Error('This mesh has no UV coordinates - cannot bake a texture onto it.')
    }

    this.on_progress(`Loading ${view_image_urls.length} view image(s)…`)
    const view_textures = await Promise.all(
      view_image_urls.map(async (url) => {
        const texture = await this.texture_loader.loadAsync(url)
        texture.colorSpace = THREE.SRGBColorSpace
        texture.needsUpdate = true
        return texture
      })
    )

    const combined_bounds = new THREE.Box3().setFromObject(scene)

    this.on_progress(`Baking ${meshes.length} mesh(es) from ${view_angles.length} angles…`)
    for (const mesh of meshes) {
      const baked_texture = this.bake_mesh(mesh, view_textures, view_angles, combined_bounds, output_resolution)
      const new_material = new THREE.MeshStandardMaterial({ map: baked_texture })
      mesh.material = new_material
    }

    this.on_progress('Exporting textured model…')
    const exported_buffer = await new Promise<ArrayBuffer>((resolve, reject) => {
      this.gltf_exporter.parse(
        scene,
        (result) => {
          if (result instanceof ArrayBuffer) {
            resolve(result)
          } else {
            reject(new Error('Baked export did not return binary GLB data'))
          }
        },
        (error: unknown) => {
          reject(error instanceof Error ? error : new Error(String(error)))
        },
        { binary: true, onlyVisible: false, embedImages: true }
      )
    })

    this.on_progress(`DEBUG: export complete, ${exported_buffer.byteLength} bytes`)
    return URL.createObjectURL(new Blob([exported_buffer], { type: 'model/gltf-binary' }))
  }

  private bake_mesh (
    mesh: THREE.Mesh,
    view_textures: THREE.Texture[],
    view_angles: ViewAngle[],
    bounds: THREE.Box3,
    resolution: number
  ): THREE.CanvasTexture {
    const render_target = new THREE.WebGLRenderTarget(resolution, resolution, {
      depthBuffer: true,
      stencilBuffer: false
    })

    // a real Camera object is only needed to satisfy renderer.render()'s
    // signature - our shader ignores its projection entirely and uses
    // the per-view uniform matrices instead
    const dummy_camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1)

    const bake_scene = new THREE.Scene()
    const bake_mesh = new THREE.Mesh(mesh.geometry)
    bake_scene.add(bake_mesh)

    this.renderer.setRenderTarget(render_target)
    this.renderer.setClearColor(0x808080, 1)
    this.renderer.clear(true, true, true)

    for (let i = 0; i < view_angles.length; i++) {
      const { view_matrix, projection_matrix, world_position } = this.compute_view_matrices(view_angles[i], bounds)

      const material = new THREE.RawShaderMaterial({
        glslVersion: THREE.GLSL3,
        vertexShader: UNFOLD_VERTEX_SHADER,
        fragmentShader: UNFOLD_FRAGMENT_SHADER,
        uniforms: {
          cameraViewMatrix: { value: view_matrix },
          cameraProjectionMatrix: { value: projection_matrix },
          cameraWorldPosition: { value: world_position },
          viewImage: { value: view_textures[i] }
        },
        side: THREE.DoubleSide,
        depthTest: true,
        depthWrite: true
      })

      bake_mesh.material = material
      this.renderer.render(bake_scene, dummy_camera)
      material.dispose()
    }

    this.renderer.setRenderTarget(null)

    // read the render target back into a canvas so it can be used as a
    // normal texture and survives GLTFExporter's embedImages step
    const pixel_buffer = new Uint8Array(resolution * resolution * 4)
    this.renderer.readRenderTargetPixels(render_target, 0, 0, resolution, resolution, pixel_buffer)
    render_target.dispose()

    const canvas = document.createElement('canvas')
    canvas.width = resolution
    canvas.height = resolution
    const context = canvas.getContext('2d')
    if (context === null) {
      throw new Error('Could not get 2D canvas context to read back baked texture')
    }
    const image_data = context.createImageData(resolution, resolution)

    // WebGL render targets are bottom-up; canvas ImageData is top-down
    for (let y = 0; y < resolution; y++) {
      const src_row = resolution - 1 - y
      const src_start = src_row * resolution * 4
      const dst_start = y * resolution * 4
      image_data.data.set(pixel_buffer.subarray(src_start, src_start + resolution * 4), dst_start)
    }
    context.putImageData(image_data, 0, 0)

    const canvas_texture = new THREE.CanvasTexture(canvas)
    canvas_texture.colorSpace = THREE.SRGBColorSpace
    canvas_texture.flipY = false
    canvas_texture.needsUpdate = true
    return canvas_texture
  }

  private compute_view_matrices (
    angle: ViewAngle,
    bounds: THREE.Box3
  ): { view_matrix: THREE.Matrix4, projection_matrix: THREE.Matrix4, world_position: THREE.Vector3 } {
    const center = new THREE.Vector3()
    bounds.getCenter(center)
    const size = new THREE.Vector3()
    bounds.getSize(size)
    const radius = Math.max(size.length() * 0.5, 0.01)

    const fov_deg = 40
    const distance = (radius / Math.sin(THREE.MathUtils.degToRad(fov_deg / 2))) * 1.15

    const azimuth = THREE.MathUtils.degToRad(angle.azimuth_deg)
    const elevation = THREE.MathUtils.degToRad(angle.elevation_deg)

    const camera_position = new THREE.Vector3(
      center.x + distance * Math.cos(elevation) * Math.sin(azimuth),
      center.y + distance * Math.sin(elevation),
      center.z + distance * Math.cos(elevation) * Math.cos(azimuth)
    )

    const camera = new THREE.PerspectiveCamera(fov_deg, 1, Math.max(distance * 0.1, 0.001), distance * 3)
    camera.position.copy(camera_position)
    camera.lookAt(center)
    camera.updateMatrixWorld(true)
    camera.updateProjectionMatrix()

    return {
      view_matrix: camera.matrixWorldInverse.clone(),
      projection_matrix: camera.projectionMatrix.clone(),
      world_position: camera_position
    }
  }
}
