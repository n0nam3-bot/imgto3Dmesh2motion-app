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
 * camera saw that surface most directly.
 *
 * CONFIRMED FIXES ALONG THE WAY (verified against real exported output,
 * not guessed):
 *   1. The render copy of the mesh needs the ORIGINAL mesh's world
 *      transform copied onto it - otherwise it renders at the origin
 *      while cameras are positioned based on the real mesh location,
 *      and every fragment fails (confirmed: produced solid clear-color
 *      output, zero coverage).
 *   2. Must use THREE.ShaderMaterial, not RawShaderMaterial - the shader
 *      relies on Three.js's automatic position/uv/normal/modelMatrix
 *      injections, which RawShaderMaterial does not provide (confirmed:
 *      this also produced solid clear-color output - the shader almost
 *      certainly never compiled, silently).
 *   3. View images must be downloaded to a local blob before use as a
 *      texture, not loaded directly from the remote (cross-origin) URL -
 *      a texture without proper CORS headers can still display normally
 *      but silently return corrupted data when read back via
 *      readRenderTargetPixels (exactly what baking needs to do).
 *   4. Not every part of the mesh is visible from any of the 6 views -
 *      confirmed by extracting and inspecting the actual baked texture
 *      from a real export: many small UV chart islands show correct,
 *      recognizable image content, but large areas remain at the
 *      uncovered clear color, causing a patchy/incomplete look on the
 *      3D model. Fixed with a dilation/gap-fill pass (see
 *      dilate_uncovered_pixels) that spreads real coverage into nearby
 *      uncovered pixels, standard practice in texture baking pipelines.
 *
 * MAIN REMAINING CALIBRATION RISK: the exact camera distance/FOV/
 * elevation used for the 6 views aren't published by the Space - only
 * the azimuth angles were recoverable from their source
 * (camera_azimuth_deg=[x-90 for x in [0,90,180,270,180,180]] ->
 * [-90,0,90,180,90,90]). The repeated 90 for views 4-5 strongly suggests
 * those are elevation-varied (top/bottom) rather than distinct azimuths;
 * DEFAULT_VIEW_ANGLES below is a reasonable guess at that split, not a
 * confirmed value.
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
uniform bool debugVisualizeUV;
in vec2 vScreenUV;
in float vFacing;
out vec4 outColor;

void main() {
  if (vFacing <= 0.05) discard;
  if (vScreenUV.x < 0.0 || vScreenUV.x > 1.0 || vScreenUV.y < 0.0 || vScreenUV.y > 1.0) discard;

  if (debugVisualizeUV) {
    // shows the raw computed screen-space UV as color instead of
    // sampling the image - a smooth gradient here means the projection
    // math is correct and the bug is in texture sampling/readback;
    // chaotic/noisy output here means the projection math itself is broken
    outColor = vec4(vScreenUV, 0.0, 1.0);
  } else {
    // force alpha=1 on every real baked pixel - this is how coverage
    // gets tracked (clear color below uses alpha=0) so the gap-fill
    // pass afterward knows exactly which pixels were actually written
    outColor = vec4(texture(viewImage, vScreenUV).rgb, 1.0);
  }
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
    output_resolution: number = 1024,
    debug_single_view_index?: number,
    debug_visualize_uv: boolean = false
  ): Promise<string> {
    if (view_image_urls.length !== view_angles.length) {
      throw new Error(
        `Got ${view_image_urls.length} view image(s) but ${view_angles.length} view angle(s) - these must match.`
      )
    }
    if (debug_single_view_index !== undefined) {
      this.on_progress(`DEBUG: isolating view index ${debug_single_view_index} only`)
      view_image_urls = [view_image_urls[debug_single_view_index]]
      view_angles = [view_angles[debug_single_view_index]]
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
        // download to a local blob first, rather than loading the remote
        // URL directly into a texture - a cross-origin texture without
        // proper CORS headers on the remote server would still DISPLAY
        // fine, but reading its rendered pixels back out (exactly what
        // baking needs) can silently return corrupted data instead of a
        // clear error. A local blob: URL can never be cross-origin-tainted.
        const image_response = await fetch(url)
        if (!image_response.ok) {
          throw new Error(`Failed to download view image (HTTP ${image_response.status}): ${url}`)
        }
        const image_blob = await image_response.blob()
        const local_url = URL.createObjectURL(image_blob)
        const texture = await this.texture_loader.loadAsync(local_url)
        texture.colorSpace = THREE.SRGBColorSpace
        texture.needsUpdate = true
        return texture
      })
    )

    const combined_bounds = new THREE.Box3().setFromObject(scene)

    this.on_progress(`Baking ${meshes.length} mesh(es) from ${view_angles.length} angle(s)…`)
    for (const mesh of meshes) {
      const baked_texture = this.bake_mesh(
        mesh, view_textures, view_angles, combined_bounds, output_resolution, debug_visualize_uv
      )
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
    resolution: number,
    debug_visualize_uv: boolean = false
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
    // CRITICAL: without this, bake_mesh renders at identity transform
    // (origin, no rotation/scale) while the cameras below are positioned
    // based on the ORIGINAL mesh's world-space bounds - every vertex
    // would land nowhere near where any camera is looking, and every
    // fragment would fail the screen-bounds/facing test, producing
    // exactly the "entirely flat clear color, zero texture" result this
    // fixes (confirmed, not hypothetical)
    mesh.updateWorldMatrix(true, false)
    bake_mesh.matrix.copy(mesh.matrixWorld)
    bake_mesh.matrixWorld.copy(mesh.matrixWorld)
    bake_mesh.matrixAutoUpdate = false
    bake_mesh.matrixWorldAutoUpdate = false
    bake_scene.add(bake_mesh)

    this.renderer.setRenderTarget(render_target)
    this.renderer.setClearColor(0x808080, 0) // alpha=0 marks "not yet covered" for the gap-fill pass below
    this.renderer.clear(true, true, true)

    for (let i = 0; i < view_angles.length; i++) {
      const { view_matrix, projection_matrix, world_position } = this.compute_view_matrices(view_angles[i], bounds)

      const material = new THREE.ShaderMaterial({
        glslVersion: THREE.GLSL3,
        vertexShader: UNFOLD_VERTEX_SHADER,
        fragmentShader: UNFOLD_FRAGMENT_SHADER,
        uniforms: {
          cameraViewMatrix: { value: view_matrix },
          cameraProjectionMatrix: { value: projection_matrix },
          cameraWorldPosition: { value: world_position },
          viewImage: { value: view_textures[i] },
          debugVisualizeUV: { value: debug_visualize_uv }
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

    // WebGL render targets are bottom-up; canvas ImageData is top-down -
    // flip while copying into a plain array we can dilate in place
    const flipped = new Uint8ClampedArray(resolution * resolution * 4)
    for (let y = 0; y < resolution; y++) {
      const src_row = resolution - 1 - y
      const src_start = src_row * resolution * 4
      const dst_start = y * resolution * 4
      flipped.set(pixel_buffer.subarray(src_start, src_start + resolution * 4), dst_start)
    }

    this.dilate_uncovered_pixels(flipped, resolution)

    const canvas = document.createElement('canvas')
    canvas.width = resolution
    canvas.height = resolution
    const context = canvas.getContext('2d')
    if (context === null) {
      throw new Error('Could not get 2D canvas context to read back baked texture')
    }
    const image_data = new ImageData(flipped, resolution, resolution)
    context.putImageData(image_data, 0, 0)

    const canvas_texture = new THREE.CanvasTexture(canvas)
    canvas_texture.colorSpace = THREE.SRGBColorSpace
    canvas_texture.flipY = false
    canvas_texture.needsUpdate = true
    return canvas_texture
  }

  /**
   * Standard texture-baking "dilation" pass: none of the 6 views can see
   * 100% of a complex mesh's surface, so parts of the UV atlas are left
   * uncovered (marked by alpha=0, set in the shader/clear color above).
   * Each pass spreads every covered pixel's color one step into any
   * directly-adjacent uncovered pixel, repeated enough times to close
   * reasonably-sized gaps and chart-boundary seams. Whatever is still
   * uncovered after all iterations gets a flat fallback fill instead of
   * being left as the raw gray clear color.
   */
  private dilate_uncovered_pixels (pixels: Uint8ClampedArray, resolution: number, iterations: number = 24): void {
    const get_index = (x: number, y: number): number => (y * resolution + x) * 4

    for (let iteration = 0; iteration < iterations; iteration++) {
      const source = pixels.slice()
      let any_filled = false

      for (let y = 0; y < resolution; y++) {
        for (let x = 0; x < resolution; x++) {
          const index = get_index(x, y)
          if (source[index + 3] !== 0) {
            continue // already covered, nothing to fill
          }

          let sum_r = 0; let sum_g = 0; let sum_b = 0; let count = 0
          const neighbors = [[x - 1, y], [x + 1, y], [x, y - 1], [x, y + 1]]
          for (const [nx, ny] of neighbors) {
            if (nx < 0 || nx >= resolution || ny < 0 || ny >= resolution) {
              continue
            }
            const neighbor_index = get_index(nx, ny)
            if (source[neighbor_index + 3] !== 0) {
              sum_r += source[neighbor_index]
              sum_g += source[neighbor_index + 1]
              sum_b += source[neighbor_index + 2]
              count++
            }
          }

          if (count > 0) {
            pixels[index] = sum_r / count
            pixels[index + 1] = sum_g / count
            pixels[index + 2] = sum_b / count
            pixels[index + 3] = 255
            any_filled = true
          }
        }
      }

      if (!any_filled) {
        break // fully converged, no point continuing
      }
    }

    // anything still uncovered after all iterations (isolated islands
    // far from any real coverage) gets a flat mid-gray instead of
    // staying at alpha=0, which would otherwise show as fully
    // transparent/black in most viewers
    for (let i = 0; i < pixels.length; i += 4) {
      if (pixels[i + 3] === 0) {
        pixels[i] = 128
        pixels[i + 1] = 128
        pixels[i + 2] = 128
        pixels[i + 3] = 255
      }
    }
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
