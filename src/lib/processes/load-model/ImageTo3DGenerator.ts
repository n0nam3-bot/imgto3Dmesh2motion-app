import { Client } from '@gradio/client'

/**
 * Generates a 3D model (GLB) from a single 2D image by calling the free,
 * publicly-hosted "tencent/Hunyuan3D-2" Hugging Face Space.
 *
 * The endpoint name and parameter order below were confirmed by reading
 * the Space's own gradio_app.py (huggingface.co/spaces/tencent/Hunyuan3D-2
 * -> Files -> gradio_app.py). Two relevant endpoints exist on that Space:
 *   - /shape_generation  -> untextured (gray) mesh, ~40s GPU budget (faster, more reliable)
 *   - /generation_all    -> textured mesh, ~90s GPU budget (slower, prettier,
 *                           more likely to hit free ZeroGPU queue limits/timeouts)
 * Defaulting to /generation_all now for textured results. If it times out
 * or fails often on the free queue, call set_space(id, '/shape_generation')
 * to fall back to the faster untextured endpoint.
 *
 * NOTE: this is a research demo, not a stable public API. Tencent can
 * change the Space's code at any time, which would require updating the
 * parameter list below to match. If generation starts failing, check
 * huggingface.co/spaces/tencent/Hunyuan3D-2/blob/main/gradio_app.py again
 * for the current function signature.
 */
export class ImageTo3DGenerator {
  private space_id: string = 'tencent/Hunyuan3D-2'
  private api_name: string = '/generation_all'

  private on_progress: (status: string) => void = () => {}

  public set_progress_callback (callback: (status: string) => void): void {
    this.on_progress = callback
  }

  public set_space (space_id: string, api_name?: string): void {
    this.space_id = space_id
    if (api_name !== undefined) {
      this.api_name = api_name
    }
  }

  /**
   * Sends an image file to the configured Hugging Face Space and resolves
   * with a blob: URL pointing to the generated GLB file, ready to be
   * passed straight into StepLoadModel.load_model_file(url, 'glb').
   */
  public async generate_from_image (image_file: File): Promise<string> {
    this.on_progress('Connecting to generation service…')
    const client = await Client.connect(this.space_id)

    this.on_progress('Uploading image and generating textured 3D model… free queues can take 1-3min')

    // positional args match shape_generation()'s signature in gradio_app.py:
    // (caption, image, mv_image_front, mv_image_back, mv_image_left, mv_image_right,
    //  steps, guidance_scale, seed, octree_resolution, check_box_rembg, num_chunks, randomize_seed)
    const result = await client.predict(this.api_name, [
      null, // caption - unused, we're doing image mode
      image_file, // image
      null, null, null, null, // multi-view images - unused
      30, // steps
      5.0, // guidance_scale
      Math.floor(Math.random() * 1e7), // seed
      384, // octree_resolution - bumped up from 256 for finer mesh detail
      true, // check_box_rembg - auto remove background, important for clean results
      8000, // num_chunks
      true // randomize_seed
    ])

    const glb_path = this.extract_glb_url(result.data)

    if (glb_path === null) {
      const raw_preview = JSON.stringify(result.data, null, 2).slice(0, 500)
      throw new Error(
        'No GLB file found in the generation response. Raw response ' +
        `(screenshot this and send it back): ${raw_preview}`
      )
    }

    // some Gradio versions return a full https URL, others a path relative
    // to the Space's own server - resolve against the Space origin if needed
    const client_config = (client as unknown as { config?: { root?: string } }).config
    const space_root = client_config?.root ?? ''
    const glb_url = glb_path.startsWith('http') ? glb_path : `${space_root}${glb_path}`

    this.on_progress('Downloading generated model…')
    const glb_response = await fetch(glb_url)
    if (!glb_response.ok) {
      throw new Error(`Failed to download generated model (HTTP ${glb_response.status})`)
    }
    const glb_blob = await glb_response.blob()

    return URL.createObjectURL(glb_blob)
  }

  /**
   * Gradio endpoints return an array of outputs. File-type outputs
   * usually come back as objects with a `url` (or `path`) property
   * (sometimes wrapped in a gr.update() `.value`), but can also come
   * back as a plain string path/URL depending on the Space's Gradio
   * version. /generation_all returns BOTH an untextured and a textured
   * mesh - this collects every GLB found and prefers one whose filename
   * suggests it's the textured version.
   */
  private extract_glb_url (data: unknown): string | null {
    if (!Array.isArray(data)) {
      return null
    }

    const found_glb_urls: string[] = []

    for (const item of data) {
      if (typeof item === 'string' && item.toLowerCase().endsWith('.glb')) {
        found_glb_urls.push(item)
        continue
      }

      if (item !== null && typeof item === 'object') {
        // gr.update(value=...) wraps the real file data under `.value`
        const unwrapped = ('value' in item && item.value !== null && typeof item.value === 'object')
          ? item.value
          : item

        const candidate = unwrapped as { url?: string, path?: string }
        const url = candidate.url ?? candidate.path
        if (typeof url === 'string' && url.toLowerCase().endsWith('.glb')) {
          found_glb_urls.push(url)
        }
      }
    }

    if (found_glb_urls.length === 0) {
      return null
    }

    const textured_match = found_glb_urls.find((url) => url.toLowerCase().includes('textured'))
    return textured_match ?? found_glb_urls[0]
  }
}
