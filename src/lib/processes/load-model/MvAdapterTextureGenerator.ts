import { Client, handle_file } from '@gradio/client'

/**
 * Calls VAST-AI's "MV-Adapter-Img2Texture" Hugging Face Space to
 * generate 6 synthesized multi-view images of a mesh, textured according
 * to a reference image. Confirmed reliable (works consistently, both via
 * this integration and the Space's own web UI).
 *
 * The Space's own second step (baking those views onto the mesh into a
 * final GLB, /run_texturing) was confirmed BROKEN - fails with a ZeroGPU
 * worker ValueError even through the Space's own UI, not just API calls.
 * generate_views_only() below stops after the working half of the
 * pipeline; MultiViewTextureBaker.ts does the baking locally instead,
 * bypassing the broken server-side step entirely.
 *
 * generate_texture_full_pipeline() (the original two-step version) is
 * kept for reference / in case the Space's texturing step gets fixed on
 * their end in the future, but generate_views_only() is the reliable
 * path.
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
   * Resolves with an array of 6 image URLs (in the same view order as
   * MultiViewTextureBaker's DEFAULT_VIEW_ANGLES), ready to feed into
   * MultiViewTextureBaker.bake(). This is the confirmed-reliable half of
   * the pipeline.
   */
  public async generate_views_only (mesh_file: File, image_file: File, prompt: string): Promise<string[]> {
    const client = await this.connect_and_start_session()

    const seed = Math.floor(Math.random() * 2_000_000_000)

    this.on_progress('Generating synthesized views… this can take 1-3min on the free queue')
    let mvadapter_result: Awaited<ReturnType<typeof client.predict>>
    try {
      mvadapter_result = await this.with_timeout(
        client.predict('/run_mvadapter', [
          mesh_file,
          prompt.trim().length > 0 ? prompt.trim() : 'high quality',
          image_file,
          seed,
          3.0,
          25,
          1.0
        ]),
        180_000,
        'Generating views timed out after 180s - the free queue is likely very busy right now.'
      )
    } catch (error: unknown) {
      throw new Error(this.describe_unknown_error(error))
    }
    this.on_progress(`DEBUG: views generated, ${JSON.stringify(mvadapter_result.data).slice(0, 150)}`)

    const raw_gallery = mvadapter_result.data[0]
    const view_urls = this.extract_gallery_urls(raw_gallery)
    if (view_urls.length === 0) {
      const raw_preview = JSON.stringify(mvadapter_result.data, null, 2).slice(0, 500)
      throw new Error(`No view images found in the response. Raw response: ${raw_preview}`)
    }

    const client_config = (client as unknown as { config?: { root?: string } }).config
    const space_root = client_config?.root ?? ''
    return view_urls.map((url) => (url.startsWith('http') ? url : `${space_root}${url}`))
  }

  /**
   * The original full two-step pipeline (views + server-side baking).
   * Kept for reference - the baking step (/run_texturing) was confirmed
   * broken on the Space's own end as of this writing. Prefer
   * generate_views_only() + MultiViewTextureBaker instead.
   */
  public async generate_texture_full_pipeline (mesh_file: File, image_file: File, prompt: string): Promise<string> {
    const client = await this.connect_and_start_session()
    const seed = Math.floor(Math.random() * 2_000_000_000)

    this.on_progress('Generating synthesized views (step 1 of 2)… this can take 1-3min on the free queue')
    let mvadapter_result: Awaited<ReturnType<typeof client.predict>>
    try {
      mvadapter_result = await this.with_timeout(
        client.predict('/run_mvadapter', [
          mesh_file,
          prompt.trim().length > 0 ? prompt.trim() : 'high quality',
          image_file,
          seed,
          3.0,
          25,
          1.0
        ]),
        180_000,
        'Step 1 (generating views) timed out after 180s.'
      )
    } catch (error: unknown) {
      throw new Error(this.describe_unknown_error(error))
    }

    const raw_gallery = mvadapter_result.data[0]
    const mv_images_gallery = this.rewrap_gallery_as_file_handles(raw_gallery)

    this.on_progress('Baking texture onto the model (step 2 of 2)… known to be unreliable on the Space right now')
    let texturing_result: Awaited<ReturnType<typeof client.predict>>
    try {
      texturing_result = await this.with_timeout(
        client.predict('/run_texturing', [
          mesh_file,
          mv_images_gallery,
          true,
          true,
          4096
        ]),
        180_000,
        'Step 2 (baking texture) timed out after 180s.'
      )
    } catch (error: unknown) {
      throw new Error(this.describe_unknown_error(error))
    }

    const glb_path = this.extract_glb_url(texturing_result.data)
    if (glb_path === null) {
      const raw_preview = JSON.stringify(texturing_result.data, null, 2).slice(0, 500)
      throw new Error(`No GLB file found in the texturing response. Raw response: ${raw_preview}`)
    }

    const client_config = (client as unknown as { config?: { root?: string } }).config
    const space_root = client_config?.root ?? ''
    const glb_url = glb_path.startsWith('http') ? glb_path : `${space_root}${glb_path}`

    const glb_response = await this.with_timeout(fetch(glb_url), 60_000, 'Timed out downloading the textured model after 60s.')
    if (!glb_response.ok) {
      throw new Error(`Failed to download textured model (HTTP ${glb_response.status})`)
    }
    const glb_blob = await glb_response.blob()
    return URL.createObjectURL(glb_blob)
  }

  private async connect_and_start_session (): Promise<Client> {
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

    return client
  }

  /**
   * Extracts plain URLs (not handle_file()-wrapped - just for
   * downloading images to pass into MultiViewTextureBaker) from a
   * Gallery-shaped response.
   */
  private extract_gallery_urls (gallery: unknown): string[] {
    if (!Array.isArray(gallery)) {
      return []
    }

    const urls: string[] = []
    for (const item of gallery) {
      const image_data = Array.isArray(item) ? item[0] : item
      const unwrapped = (image_data !== null && typeof image_data === 'object' && 'image' in (image_data as object))
        ? (image_data as { image: unknown }).image
        : image_data

      if (typeof unwrapped === 'string') {
        urls.push(unwrapped)
        continue
      }
      if (unwrapped !== null && typeof unwrapped === 'object') {
        const candidate = unwrapped as { url?: string, path?: string }
        const url = candidate.url ?? candidate.path
        if (typeof url === 'string') {
          urls.push(url)
        }
      }
    }
    return urls
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

  private describe_unknown_error (error: unknown): string {
    if (error instanceof Error) {
      return error.message
    }
    if (typeof error === 'string') {
      return error
    }
    if (error !== null && typeof error === 'object') {
      const candidate = error as Record<string, unknown>
      const readable = candidate.message ?? candidate.detail ?? candidate.error
      const stage = typeof candidate.stage === 'string' ? ` (stage: ${candidate.stage})` : ''
      if (typeof readable === 'string') {
        return `${readable}${stage}`
      }
      try {
        return `Unrecognized error object: ${JSON.stringify(error).slice(0, 500)}`
      } catch {
        return 'Unrecognized error object that could not be serialized.'
      }
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
