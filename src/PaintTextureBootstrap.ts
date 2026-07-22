import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { GLTFExporter } from 'three/examples/jsm/exporters/GLTFExporter.js'

/**
 * Standalone model texture painter. Deliberately independent from
 * Mesh2MotionEngine (which drives the rigging/animation flow) - this is
 * a focused tool: load a GLB, paint on it, export a GLB back out.
 *
 * KNOWN FIRST-PASS RISK AREAS (untested against a live browser - if
 * painting appears mirrored or offset from where you click, these are
 * the first two things to check):
 *   1. UV vertical orientation: this.paint_texture.flipY is set to false
 *      to match glTF's texture convention, and paint_at_uv() maps v
 *      directly (no 1-v flip) to match. If painting appears upside-down
 *      or on the wrong side of the model, try toggling flipY and
 *      swapping to (1 - uv.y) in paint_at_uv().
 *   2. Multi-material meshes: if a mesh's `.material` is an array
 *      (multiple materials/material groups on one geometry), only the
 *      first is currently replaced with the paintable canvas texture.
 */
class PaintTextureBootstrap {
  private readonly renderer = new THREE.WebGLRenderer({ antialias: true })
  private readonly scene = new THREE.Scene()
  private readonly camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.01, 1000)
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
  private readonly undo_stack: ImageData[] = []
  private readonly max_undo_steps: number = 20

  private brush_color: string = '#c23b3b'
  private brush_size: number = 24

  private readonly dom_upload_input = document.querySelector<HTMLInputElement>('#paint-model-upload')
  private readonly dom_filename = document.querySelector<HTMLElement>('#paint-model-filename')
  private readonly dom_color = document.querySelector<HTMLInputElement>('#paint-brush-color')
  private readonly dom_size = document.querySelector<HTMLInputElement>('#paint-brush-size')
  private readonly dom_size_value = document.querySelector<HTMLElement>('#paint-brush-size-value')
  private readonly dom_undo_button = document.querySelector<HTMLButtonElement>('#paint-undo-button')
  private readonly dom_clear_button = document.querySelector<HTMLButtonElement>('#paint-clear-button')
  private readonly dom_download_button = document.querySelector<HTMLButtonElement>('#paint-download-button')
  private readonly dom_status = document.querySelector<HTMLElement>('#paint-status')

  constructor () {
    const context = this.paint_canvas.getContext('2d')
    if (context === null) {
      throw new Error('Could not get 2D canvas context for painting')
    }
    this.paint_context = context
    this.paint_canvas.width = 1024
    this.paint_canvas.height = 1024

    this.controls = new OrbitControls(this.camera, this.renderer.domElement)

    this.animate = this.animate.bind(this)
    this.on_resize = this.on_resize.bind(this)

    this.setup_renderer()
    this.setup_scene_lighting()
    this.setup_dom_listeners()
    this.animate()
  }

  private setup_renderer (): void {
    this.renderer.setSize(window.innerWidth, window.innerHeight)
    this.renderer.setPixelRatio(window.devicePixelRatio)
    this.renderer.domElement.style.touchAction = 'none' // matches existing app pattern for touch/orbit
    document.body.appendChild(this.renderer.domElement)

    this.camera.position.set(0, 1.4, 3)
    this.controls.target.set(0, 1, 0)
    this.controls.enableDamping = true
    this.controls.update()

    window.addEventListener('resize', this.on_resize)
  }

  private setup_scene_lighting (): void {
    const ambient_light = new THREE.AmbientLight(0xffffff, 1.3)
    this.scene.add(ambient_light)

    const directional_light = new THREE.DirectionalLight(0xffffff, 1.6)
    directional_light.position.set(2, 4, 3)
    this.scene.add(directional_light)

    const fill_light = new THREE.DirectionalLight(0xffffff, 0.6)
    fill_light.position.set(-2, 1, -3)
    this.scene.add(fill_light)
  }

  private setup_dom_listeners (): void {
    this.dom_upload_input?.addEventListener('change', () => { this.on_model_selected() })

    this.dom_color?.addEventListener('input', () => {
      this.brush_color = this.dom_color?.value ?? this.brush_color
    })

    this.dom_size?.addEventListener('input', () => {
      this.brush_size = Number(this.dom_size?.value ?? this.brush_size)
      if (this.dom_size_value !== null) {
        this.dom_size_value.textContent = String(this.brush_size)
      }
    })

    this.dom_undo_button?.addEventListener('click', () => { this.undo() })
    this.dom_clear_button?.addEventListener('click', () => { this.clear_to_color() })
    this.dom_download_button?.addEventListener('click', () => { this.export_model() })

    const canvas_element = this.renderer.domElement
    canvas_element.addEventListener('pointerdown', (event) => { this.on_pointer_down(event) })
    canvas_element.addEventListener('pointermove', (event) => { this.on_pointer_move(event) })
    window.addEventListener('pointerup', () => { this.on_pointer_up() })
    window.addEventListener('pointercancel', () => { this.on_pointer_up() })
  }

  private on_model_selected (): void {
    const file = this.dom_upload_input?.files?.[0]
    if (file === undefined) {
      return
    }

    const object_url = URL.createObjectURL(file)
    this.set_status('Loading model…')

    this.gltf_loader.load(
      object_url,
      (gltf) => {
        if (this.loaded_scene !== null) {
          this.scene.remove(this.loaded_scene)
        }

        this.loaded_scene = gltf.scene
        this.scene.add(this.loaded_scene)

        this.setup_paintable_mesh()
        this.frame_camera_to_object(this.loaded_scene)

        if (this.dom_filename !== null) {
          this.dom_filename.textContent = file.name
        }
      },
      undefined,
      (error: unknown) => {
        const message = error instanceof Error ? error.message : String(error)
        this.set_status(`Failed to load model: ${message}`)
      }
    )
  }

  private setup_paintable_mesh (): void {
    if (this.loaded_scene === null) {
      return
    }

    let found_mesh: THREE.Mesh | null = null
    this.loaded_scene.traverse((child) => {
      if (found_mesh === null && child instanceof THREE.Mesh && child.geometry.attributes.uv !== undefined) {
        found_mesh = child
      }
    })

    if (found_mesh === null) {
      this.paintable_mesh = null
      this.set_status(
        'This model has no UV coordinates, so it can\'t be painted on directly. ' +
        'Try a model that already has texture/UVs (most Hunyuan3D-generated or ' +
        'reference models should work).'
      )
      return
    }

    this.paintable_mesh = found_mesh

    // seed the paint canvas with the mesh's existing texture if it has one,
    // otherwise a flat fill so there's something visible to paint over
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
    this.paint_texture.flipY = false // see class-level comment: matches glTF UV convention

    const new_material = new THREE.MeshStandardMaterial({ map: this.paint_texture })
    found_mesh.material = new_material

    this.undo_stack.length = 0
    this.push_undo_snapshot()

    this.set_status('Model loaded. Drag directly on the model to paint.')
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
    const hit = this.raycast_from_event(event)
    if (hit === null) {
      return // clicked empty space - let OrbitControls handle orbiting normally
    }

    event.preventDefault()
    this.is_painting = true
    this.controls.enabled = false
    this.push_undo_snapshot()
    this.paint_at_hit(hit)
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
    const pixel_y = hit.uv.y * this.paint_canvas.height // see class-level comment on flipY

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

  private undo (): void {
    const snapshot = this.undo_stack.pop()
    if (snapshot === undefined || this.paint_texture === null) {
      return
    }
    this.paint_context.putImageData(snapshot, 0, 0)
    this.paint_texture.needsUpdate = true
  }

  private clear_to_color (): void {
    if (this.paint_texture === null) {
      return
    }
    this.push_undo_snapshot()
    this.paint_context.fillStyle = this.dom_color?.value ?? this.brush_color
    this.paint_context.fillRect(0, 0, this.paint_canvas.width, this.paint_canvas.height)
    this.paint_texture.needsUpdate = true
  }

  private export_model (): void {
    if (this.loaded_scene === null) {
      this.set_status('No model loaded yet.')
      return
    }

    this.set_status('Exporting…')

    this.gltf_exporter.parse(
      this.loaded_scene,
      (result) => {
        if (!(result instanceof ArrayBuffer)) {
          this.set_status('Export did not return binary GLB data.')
          return
        }

        const blob = new Blob([result], { type: 'model/gltf-binary' })
        const download_link = document.createElement('a')
        download_link.href = URL.createObjectURL(blob)
        download_link.download = 'painted-model.glb'
        document.body.appendChild(download_link)
        download_link.click()
        document.body.removeChild(download_link)

        this.set_status('Downloaded painted-model.glb')
      },
      (error: unknown) => {
        const message = error instanceof Error ? error.message : String(error)
        this.set_status(`Export failed: ${message}`)
      },
      { binary: true, onlyVisible: false, embedImages: true }
    )
  }

  private set_status (message: string): void {
    if (this.dom_status !== null) {
      this.dom_status.textContent = message
    }
  }

  private on_resize (): void {
    this.camera.aspect = window.innerWidth / window.innerHeight
    this.camera.updateProjectionMatrix()
    this.renderer.setSize(window.innerWidth, window.innerHeight)
  }

  private animate (): void {
    requestAnimationFrame(this.animate)
    this.controls.update()
    this.renderer.render(this.scene, this.camera)
  }
}

// instantiate the class to set up the whole tool
const app = new PaintTextureBootstrap()
