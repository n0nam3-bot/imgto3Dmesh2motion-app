import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { GLTFExporter } from 'three/examples/jsm/exporters/GLTFExporter.js'
import UvUnwrapWorkerConstructor from './UvUnwrapWorker.ts?worker'
import type { UvUnwrapRequest, UvUnwrapResponse } from './UvUnwrapWorker.ts'

/**
 * Embeddable texture painter. Same painting approach as the standalone
 * paint.html tool, but scoped to a caller-provided container element
 * (sized by CSS, not fullscreen) so it can be used inside an in-page
 * overlay - no separate tab/page needed.
 *
 * KNOWN FIRST-PASS RISK AREAS (see paint.html's tool for the same notes -
 * this is the same painting logic, just re-scoped):
 *   1. UV vertical orientation (paint_texture.flipY / paint_at_hit mapping)
 *   2. Multi-material meshes only get their first material replaced
 */
export class EmbeddedTexturePainter {
  private readonly renderer = new THREE.WebGLRenderer({ antialias: true })
  private readonly scene = new THREE.Scene()
  private readonly camera = new THREE.PerspectiveCamera(50, 1, 0.01, 1000)
  private readonly controls: OrbitControls
  private readonly raycaster = new THREE.Raycaster()
  private readonly pointer = new THREE.Vector2()
  private readonly gltf_loader = new GLTFLoader()
  private readonly gltf_exporter = new GLTFExporter()

  private loaded_scene: THREE.Group | null = null
  private paintable_mesh: THREE.Mesh | null = null

  private readonly paint_canvas: HTMLCanvasElement = document.createElement('canvas')
  private readonly paint_context: CanvasRenderingContext2D
  private paint_texture: THREE.CanvasTexture | null = null

  private is_painting: boolean = false
  private is_animating: boolean = false
  private readonly undo_stack: ImageData[] = []
  private readonly max_undo_steps: number = 20

  private brush_color: string = '#c23b3b'
  private brush_size: number = 24

  private readonly container: HTMLElement
  private on_status: (message: string) => void = () => {}

  constructor (container: HTMLElement) {
    this.container = container

    const context = this.paint_canvas.getContext('2d')
    if (context === null) {
      throw new Error('Could not get 2D canvas context for painting')
    }
    this.paint_context = context
    this.paint_canvas.width = 1024
    this.paint_canvas.height = 1024

    this.renderer.domElement.style.touchAction = 'none'
    this.renderer.domElement.style.width = '100%'
    this.renderer.domElement.style.height = '100%'
    this.renderer.domElement.style.display = 'block'
    this.container.appendChild(this.renderer.domElement)

    this.controls = new OrbitControls(this.camera, this.renderer.domElement)
    this.controls.enableDamping = true

    const ambient_light = new THREE.AmbientLight(0xffffff, 1.3)
    this.scene.add(ambient_light)
    const directional_light = new THREE.DirectionalLight(0xffffff, 1.6)
    directional_light.position.set(2, 4, 3)
    this.scene.add(directional_light)
    const fill_light = new THREE.DirectionalLight(0xffffff, 0.6)
    fill_light.position.set(-2, 1, -3)
    this.scene.add(fill_light)

    this.animate = this.animate.bind(this)
    this.setup_pointer_listeners()
  }

  public set_status_callback (callback: (message: string) => void): void {
    this.on_status = callback
  }

  public set_brush_color (color: string): void {
    this.brush_color = color
  }

  public set_brush_size (size: number): void {
    this.brush_size = size
  }

  /** Call after the container becomes visible, so sizing reads correctly. */
  public handle_resize (): void {
    const width = this.container.clientWidth || 1
    const height = this.container.clientHeight || 1
    this.camera.aspect = width / height
    this.camera.updateProjectionMatrix()
    this.renderer.setSize(width, height)
  }

  public start_render_loop (): void {
    if (this.is_animating) {
      return
    }
    this.is_animating = true
    this.animate()
  }

  public stop_render_loop (): void {
    this.is_animating = false
  }

  public load_model_from_url (glb_url: string): void {
    this.on_status('DEBUG: starting GLTFLoader.load()…')

    this.gltf_loader.load(
      glb_url,
      (gltf) => {
        try {
          if (this.loaded_scene !== null) {
            this.scene.remove(this.loaded_scene)
          }
          this.loaded_scene = gltf.scene
          this.scene.add(this.loaded_scene)

          let mesh_count = 0
          this.loaded_scene.traverse((child) => {
            if (child instanceof THREE.Mesh) {
              mesh_count++
            }
          })
          this.on_status(`DEBUG: GLTFLoader succeeded, ${mesh_count} mesh(es) found in scene`)

          this.frame_camera_to_object(this.loaded_scene)
          void this.setup_paintable_mesh()
        } catch (error: unknown) {
          const message = error instanceof Error ? (error.stack ?? error.message) : String(error)
          this.on_status(`DEBUG: exception in onLoad callback: ${message}`)
        }
      },
      undefined,
      (error: unknown) => {
        const message = error instanceof Error ? error.message : String(error)
        this.on_status(`DEBUG: GLTFLoader onError fired: ${message}`)
      }
    )
  }

  private async setup_paintable_mesh (): Promise<void> {
    if (this.loaded_scene === null) {
      return
    }

    let found_mesh: THREE.Mesh | null = null
    let largest_mesh: THREE.Mesh | null = null
    let largest_vertex_count = -1

    this.loaded_scene.traverse((child) => {
      if (!(child instanceof THREE.Mesh)) {
        return
      }
      if (found_mesh === null && child.geometry.attributes.uv !== undefined) {
        found_mesh = child
      }
      const vertex_count = child.geometry.attributes.position?.count ?? 0
      if (vertex_count > largest_vertex_count) {
        largest_vertex_count = vertex_count
        largest_mesh = child
      }
    })

    let used_generated_uvs = false
    if (found_mesh === null && largest_mesh !== null) {
      // no mesh had UVs at all (common for untextured/shape-only outputs,
      // which have no reason to include a texture mapping) - generate
      // real UVs with xatlas so painting works properly on it.
      this.on_status('No texture coordinates on this model - unwrapping UVs…')
      const mesh_to_unwrap = largest_mesh as THREE.Mesh
      try {
        await this.generate_real_uvs(mesh_to_unwrap.geometry)
      } catch (error: unknown) {
        console.error('xatlas UV unwrap failed, falling back to spherical projection', error)
        this.generate_spherical_uvs(mesh_to_unwrap.geometry)
      }
      found_mesh = mesh_to_unwrap
      used_generated_uvs = true
    }

    if (found_mesh === null) {
      this.paintable_mesh = null
      this.on_status('No mesh found in this model at all.')
      return
    }

    this.paintable_mesh = found_mesh

    this.paint_context.clearRect(0, 0, this.paint_canvas.width, this.paint_canvas.height)
    const existing_material = Array.isArray(found_mesh.material) ? found_mesh.material[0] : found_mesh.material
    const existing_map = (existing_material as THREE.MeshStandardMaterial)?.map

    if (existing_map?.image !== undefined && existing_map.image !== null) {
      try {
        this.paint_context.drawImage(existing_map.image, 0, 0, this.paint_canvas.width, this.paint_canvas.height)
      } catch {
        this.fill_canvas_background()
      }
    } else {
      this.fill_canvas_background()
    }

    this.paint_texture = new THREE.CanvasTexture(this.paint_canvas)
    this.paint_texture.flipY = false

    const new_material = new THREE.MeshStandardMaterial({ map: this.paint_texture })
    found_mesh.material = new_material

    this.undo_stack.length = 0
    this.push_undo_snapshot()

    this.on_status(
      used_generated_uvs
        ? 'UVs generated. Drag directly on the model to paint. Drag empty space to orbit.'
        : 'Drag directly on the model to paint. Drag empty space to orbit.'
    )
  }

  /**
   * Generates real UV coordinates for a mesh that has none, using xatlas
   * (the same UV-unwrapping library used by Blender's glTF exporter and
   * other production tools) - run in a Web Worker rather than on the main
   * thread, since it's synchronous CPU-bound work that can otherwise
   * freeze the entire page with no way to recover (confirmed on a real
   * AI-generated mesh). Running it in a worker also means a timeout can
   * actually terminate it if it takes too long, which isn't possible for
   * a blocking main-thread call.
   */
  private async generate_real_uvs (geometry: THREE.BufferGeometry): Promise<void> {
    const position_attribute = geometry.attributes.position
    if (position_attribute === undefined) {
      throw new Error('Geometry has no position attribute to unwrap')
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
      // unreachable in practice - the worker promise only ever resolves on
      // 'success' or 'error' - but keeps TypeScript's narrowing happy
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
    this.on_status(`DEBUG: creating worker (${request.positions.length / 3} verts, ${request.indices?.length ?? 'none'} indices)`)
    const worker = new UvUnwrapWorkerConstructor()
    this.on_status('DEBUG: worker constructed, posting message…')

    try {
      return await new Promise<UvUnwrapResponse>((resolve, reject) => {
        const timeout_handle = window.setTimeout(() => {
          reject(new Error(`UV unwrapping timed out after ${Math.round(timeout_ms / 1000)}s.`))
        }, timeout_ms)

        worker.onmessage = (event: MessageEvent<UvUnwrapResponse>) => {
          if (event.data.status === 'progress') {
            this.on_status(`DEBUG: worker progress: ${event.data.message}`)
            return
          }
          this.on_status(`DEBUG: worker responded with status=${event.data.status}`)
          window.clearTimeout(timeout_handle)
          resolve(event.data)
        }

        worker.onerror = (error: ErrorEvent) => {
          this.on_status(`DEBUG: worker.onerror fired: ${error.message} (${error.filename}:${error.lineno})`)
          window.clearTimeout(timeout_handle)
          reject(new Error(error.message))
        }

        worker.onmessageerror = () => {
          this.on_status('DEBUG: worker.onmessageerror fired (data could not be deserialized)')
        }

        const transfer_list: ArrayBuffer[] = [request.positions.buffer]
        if (request.indices !== undefined) {
          transfer_list.push(request.indices.buffer)
        }
        if (request.normals !== undefined) {
          transfer_list.push(request.normals.buffer)
        }

        worker.postMessage(request, transfer_list)
        this.on_status('DEBUG: message posted, waiting for response…')
      })
    } finally {
      worker.terminate()
    }
  }

  /**
   * Emergency fallback only - used if xatlas itself throws. A simple
   * spherical projection from the geometry's center. Expect visible
   * seams/stretching; this is not a real UV unwrap.
   */
  private generate_spherical_uvs (geometry: THREE.BufferGeometry): void {
    const position = geometry.attributes.position
    if (position === undefined) {
      return
    }

    geometry.computeBoundingSphere()
    const center = geometry.boundingSphere?.center ?? new THREE.Vector3()

    const vertex_count = position.count
    const uv_array = new Float32Array(vertex_count * 2)
    const direction = new THREE.Vector3()

    for (let i = 0; i < vertex_count; i++) {
      direction.set(
        position.getX(i) - center.x,
        position.getY(i) - center.y,
        position.getZ(i) - center.z
      )

      if (direction.lengthSq() < 1e-10) {
        direction.set(0, 1, 0) // degenerate case: vertex sits exactly at center
      } else {
        direction.normalize()
      }

      uv_array[i * 2] = 0.5 + Math.atan2(direction.z, direction.x) / (2 * Math.PI)
      uv_array[i * 2 + 1] = 0.5 - Math.asin(THREE.MathUtils.clamp(direction.y, -1, 1)) / Math.PI
    }

    geometry.setAttribute('uv', new THREE.BufferAttribute(uv_array, 2))
  }

  private fill_canvas_background (): void {
    this.paint_context.fillStyle = '#c9c9c9'
    this.paint_context.fillRect(0, 0, this.paint_canvas.width, this.paint_canvas.height)
  }

  private frame_camera_to_object (object: THREE.Object3D): void {
    const bounding_box = new THREE.Box3().setFromObject(object)
    const size = new THREE.Vector3()
    bounding_box.getSize(size)
    const center = new THREE.Vector3()
    bounding_box.getCenter(center)

    const max_dimension = Math.max(size.x, size.y, size.z, 0.01)
    const distance = max_dimension * 1.8

    this.camera.position.set(center.x, center.y + size.y * 0.15, center.z + distance)
    this.camera.near = distance / 100
    this.camera.far = distance * 100
    this.camera.updateProjectionMatrix()

    this.controls.target.copy(center)
    this.controls.update()
  }

  private setup_pointer_listeners (): void {
    const canvas_element = this.renderer.domElement
    canvas_element.addEventListener('pointerdown', (event) => { this.on_pointer_down(event) })
    canvas_element.addEventListener('pointermove', (event) => { this.on_pointer_move(event) })
    window.addEventListener('pointerup', () => { this.on_pointer_up() })
    window.addEventListener('pointercancel', () => { this.on_pointer_up() })
  }

  private raycast_from_event (event: PointerEvent): THREE.Intersection | null {
    if (this.paintable_mesh === null) {
      return null
    }

    const bounds = this.renderer.domElement.getBoundingClientRect()
    this.pointer.x = ((event.clientX - bounds.left) / bounds.width) * 2 - 1
    this.pointer.y = -((event.clientY - bounds.top) / bounds.height) * 2 + 1

    this.raycaster.setFromCamera(this.pointer, this.camera)
    const intersections = this.raycaster.intersectObject(this.paintable_mesh, false)
    return intersections.length > 0 ? intersections[0] : null
  }

  private on_pointer_down (event: PointerEvent): void {
    if (this.paintable_mesh === null) {
      this.on_status('DEBUG: no paintable mesh set (model may not have loaded, or has no UV coordinates).')
      return
    }

    const hit = this.raycast_from_event(event)
    if (hit === null) {
      this.on_status('DEBUG: pointer registered, but raycast found no surface under it.')
      return
    }
    event.preventDefault()
    this.is_painting = true
    this.controls.enabled = false
    this.push_undo_snapshot()
    this.paint_at_hit(hit)
    this.on_status(`DEBUG: hit at uv (${hit.uv?.x.toFixed(2) ?? 'none'}, ${hit.uv?.y.toFixed(2) ?? 'none'})`)
  }

  private on_pointer_move (event: PointerEvent): void {
    if (!this.is_painting) {
      return
    }
    const hit = this.raycast_from_event(event)
    if (hit !== null) {
      event.preventDefault()
      this.paint_at_hit(hit)
    }
  }

  private on_pointer_up (): void {
    this.is_painting = false
    this.controls.enabled = true
  }

  private paint_at_hit (hit: THREE.Intersection): void {
    if (hit.uv === undefined || this.paint_texture === null) {
      return
    }

    const pixel_x = hit.uv.x * this.paint_canvas.width
    const pixel_y = hit.uv.y * this.paint_canvas.height

    this.paint_context.fillStyle = this.brush_color
    this.paint_context.beginPath()
    this.paint_context.arc(pixel_x, pixel_y, this.brush_size, 0, Math.PI * 2)
    this.paint_context.fill()

    this.paint_texture.needsUpdate = true
  }

  private push_undo_snapshot (): void {
    if (this.undo_stack.length >= this.max_undo_steps) {
      this.undo_stack.shift()
    }
    this.undo_stack.push(
      this.paint_context.getImageData(0, 0, this.paint_canvas.width, this.paint_canvas.height)
    )
  }

  public undo (): void {
    const snapshot = this.undo_stack.pop()
    if (snapshot === undefined || this.paint_texture === null) {
      return
    }
    this.paint_context.putImageData(snapshot, 0, 0)
    this.paint_texture.needsUpdate = true
  }

  public clear_to_color (color: string): void {
    if (this.paint_texture === null) {
      return
    }
    this.push_undo_snapshot()
    this.paint_context.fillStyle = color
    this.paint_context.fillRect(0, 0, this.paint_canvas.width, this.paint_canvas.height)
    this.paint_texture.needsUpdate = true
  }

  /** Exports the current (painted) scene as a new GLB blob: URL. */
  public async export_glb_blob_url (): Promise<string> {
    if (this.loaded_scene === null) {
      throw new Error('No model loaded in the paint view.')
    }

    const exported_buffer = await new Promise<ArrayBuffer>((resolve, reject) => {
      this.gltf_exporter.parse(
        this.loaded_scene as THREE.Group,
        (result) => {
          if (result instanceof ArrayBuffer) {
            resolve(result)
          } else {
            reject(new Error('Export did not return binary GLB data'))
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

  private animate (): void {
    if (!this.is_animating) {
      return
    }
    requestAnimationFrame(this.animate)
    this.controls.update()
    this.renderer.render(this.scene, this.camera)
  }
}
