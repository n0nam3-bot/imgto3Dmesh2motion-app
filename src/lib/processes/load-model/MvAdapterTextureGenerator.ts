import { Client, handle_file } from '@gradio/client'

/**
 * Generates real, plausible all-around texture for an existing GLB mesh
 * from a single reference image, using VAST-AI's "MV-Adapter-Img2Texture"
 * Hugging Face Space. Unlike TextureApplier (simple front-projection),
 * this actually synthesizes texture for unseen angles using a multi-view
 * diffusion model - genuinely different technology, verified against the
 * Space's real source (huggingface.co/spaces/VAST-AI/MV-Adapter-Img2Texture
 * -> Files -> app.py) rather than guessed.
 *
 * This is a two-step pipeline on the Space's side:
 *   1. /run_mvadapter - generates 6 synthesized view images of the mesh
 *      textured according to the reference image (~90s GPU budget)
 *   2. /run_texturing - bakes those 6 views onto the mesh's UVs, producing
 *      the final textured GLB (~90s GPU budget)
 * Both steps run on the free ZeroGPU queue, so total time can run from
 * under a minute to several minutes depending on queue load.
 *
 * The Space also has a demo.load(start_session) handler that creates a
 * per-session working directory - normally triggered by a browser
 * loading the page, but never fired for API-only access, so it's called
 * explicitly here first (confirmed necessary: run_texturing writes into
 * that directory and fails without it existing).
 *
 * NOTE: this is a research demo, not a stable public API - the Space
 * owner can change the code at any time.
 */
export class MvAdapterTextureGenerator {
  private readonly space_id: string = 'VAST-AI/MV-Adapter-Img2Texture'
  private hf_token: string | undefined

  private on_progress: (message: string) => void = () => {}

  public set_progress_callback (callback: (message: string) => void): void {
    this.on_progress = callback
  }

  public set_hf_token (token: string | undefined): void {
    this.hf_token = (token !== undefined && token.trim().length > 0) ? token.trim() : undefined
  }

  /**
   * Resolves with a blob: URL for the textured GLB.
   */
  public async generate_texture (mesh_file: File, image_file: File, prompt: string): Promise<string> {
    this.on_progress('DEBUG: connecting to MV-Adapter service…')
    const connect_options = this.hf_token !== undefined ? { token: this.hf_token as `hf_${string}` } : undefined
    const client = await Client.connect(this.space_id, connect_options)
    this.on_progress('DEBUG: connected')

    try {
      await this.with_timeout(client.predict('/start_session', []), 30_000, 'Timed out starting session after 30s.')
      this.on_progress('DEBUG: session started')
    } catch (error: unknown) {
      this.on_progress(`DEBUG: start_session call failed (continuing anyway): ${this.describe_unknown_error(error)}`)
    }

    const seed = Math.floor(Math.random() * 2_000_000_000)

    this.on_progress('Generating synthesized views (step 1 of 2)… this can take 1-3min on the free queue')
    let mvadapter_result: Awaited<ReturnType<typeof client.predict>>
    try {
      mvadapter_result = await this.with_timeout(
        client.predict('/run_mvadapter', [
          mesh_file, // input_mesh
          prompt.trim().length > 0 ? prompt.trim() : 'high quality', // prompt
          image_file, // image_prompt
          seed, // seed
          3.0, // guidance_scale (Space default)
          25, // num_inference_steps (Space default)
          1.0 // reference_conditioning_scale (Space default)
        ]),
        180_000,
        'Step 1 (generating views) timed out after 180s - the free queue is likely very busy right now.'
      )
    } catch (error: unknown) {
      throw new Error(this.describe_unknown_error(error))
    }
    this.on_progress(`DEBUG: step 1 done, ${JSON.stringify(mvadapter_result.data).slice(0, 150)}`)

    const raw_gallery = mvadapter_result.data[0]
    const mv_images_gallery = this.rewrap_gallery_as_file_handles(raw_gallery)
    this.on_progress(`DEBUG: rewrapped ${mv_images_gallery.length} view image(s) for step 2`)

    this.on_progress('Baking texture onto the model (step 2 of 2)… this can take 1-3min on the free queue')
    let texturing_result: Awaited<ReturnType<typeof client.predict>>
    try {
      texturing_result = await this.with_timeout(
        client.predict('/run_texturing', [
          mesh_file, // input_mesh - same original mesh file again
          mv_images_gallery, // mv_result - fed back from step 1's output, rewrapped
          true, // uv_unwarp (Space default)
          // Space's own mesh-cleanup step, meant for exactly this kind of
          // input (AI-generated geometry can be non-manifold/messy) -
          // was previously off (matching the UI default), now on since
          // that's a well-justified thing to try for our specific
          // use case rather than the Space's general-purpose default
          true, // preprocess_mesh
          4096 // uv_size (Space default)
        ]),
        180_000,
        'Step 2 (baking texture) timed out after 180s - the free queue is likely very busy right now.'
      )
    } catch (error: unknown) {
      throw new Error(this.describe_unknown_error(error))
    }
    this.on_progress('DEBUG: step 2 done')

    const glb_path = this.extract_glb_url(texturing_result.data)
    if (glb_path === null) {
      const raw_preview = JSON.stringify(texturing_result.data, null, 2).slice(0, 500)
      throw new Error(
        'No GLB file found in the texturing response. Raw response ' +
        `(screenshot this and send it back): ${raw_preview}`
      )
    }

    const client_config = (client as unknown as { config?: { root?: string } }).config
    const space_root = client_config?.root ?? ''
    const glb_url = glb_path.startsWith('http') ? glb_path : `${space_root}${glb_path}`

    this.on_progress('Downloading textured model…')
    const glb_response = await this.with_timeout(fetch(glb_url), 60_000, 'Timed out downloading the textured model after 60s.')
    if (!glb_response.ok) {
      throw new Error(`Failed to download textured model (HTTP ${glb_response.status})`)
    }
    const glb_blob = await glb_response.blob()

    return URL.createObjectURL(glb_blob)
  }

  private rewrap_gallery_as_file_handles (gallery: unknown): unknown[] {
    if (!Array.isArray(gallery)) {
      return []
    }

    const wrapped: unknown[] = []

    for (const item of gallery) {
      const image_data = Array.isArray(item) ? item[0] : item

      const unwrapped = (image_data !== null && typeof image_data === 'object' && 'image' in (image_data as object))
        ? (image_data as { image: unknown }).image
        : image_data

      if (typeof unwrapped === 'string') {
        wrapped.push([handle_file(unwrapped), null])
        continue
      }

      if (unwrapped !== null && typeof unwrapped === 'object') {
        const candidate = unwrapped as { url?: string, path?: string }
        const url = candidate.url ?? candidate.path
        if (typeof url === 'string') {
          wrapped.push([handle_file(url), null])
        }
      }
    }

    return wrapped
  }

  private async with_timeout<T> (promise: Promise<T>, timeout_ms: number, timeout_message: string): Promise<T> {
    let timeout_handle: number | undefined
    const timeout_promise = new Promise<never>((_, reject) => {
      timeout_handle = window.setTimeout(() => { reject(new Error(timeout_message)) }, timeout_ms)
    })

    try {
      return await Promise.race([promise, timeout_promise])
    } finally {
      if (timeout_handle !== undefined) {
        window.clearTimeout(timeout_handle)
      }
    }
  }

  /**
   * Digs through every plausible location for a human-readable message,
   * since "An error occurred" (Gradio's sanitized default for unhandled
   * server exceptions) has shown up with no further detail in the
   * top-level fields checked before - this also dumps the FULL raw
   * object as a last resort so nothing is silently lost.
   */
  private describe_unknown_error (error: unknown): string {
    if (error instanceof Error) {
      return error.message
    }
    if (typeof error === 'string') {
      return error
    }
    if (error !== null && typeof error === 'object') {
      const candidate = error as Record<string, unknown>
      const direct_message = candidate.message ?? candidate.detail ?? candidate.error
      const nested_message = (candidate.data !== null && typeof candidate.data === 'object')
        ? (candidate.data as Record<string, unknown>).message ?? (candidate.data as Record<string, unknown>).error
        : undefined
      const body_message = (candidate.body !== null && typeof candidate.body === 'object')
        ? (candidate.body as Record<string, unknown>).message ?? (candidate.body as Record<string, unknown>).error
        : undefined

      const readable = direct_message ?? nested_message ?? body_message
      const stage = typeof candidate.stage === 'string' ? ` (stage: ${candidate.stage})` : ''

      let full_dump = ''
      try {
        full_dump = ` | full error object: ${JSON.stringify(error, Object.getOwnPropertyNames(error)).slice(0, 800)}`
      } catch {
        full_dump = ' | (error object could not be serialized)'
      }

      if (typeof readable === 'string') {
        return `${readable}${stage}${full_dump}`
      }
      return `Unrecognized error object${stage}${full_dump}`
    }
    return String(error)
  }

  private extract_glb_url (data: unknown): string | null {
    if (!Array.isArray(data)) {
      return null
    }

    for (const item of data) {
      if (typeof item === 'string' && item.toLowerCase().endsWith('.glb')) {
        return item
      }
      if (item !== null && typeof item === 'object') {
        const unwrapped = ('value' in item && item.value !== null && typeof item.value === 'object')
          ? item.value
          : item
        const candidate = unwrapped as { url?: string, path?: string }
        const url = candidate.url ?? candidate.path
        if (typeof url === 'string' && url.toLowerCase().endsWith('.glb')) {
          return url
        }
      }
    }

    return null
  }
}
