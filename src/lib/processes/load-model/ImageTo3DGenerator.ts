import { Client, handle_file } from '@gradio/client'

export type ImageTo3DProvider = 'hunyuan3d2' | 'triposr'

/**
 * Generates a 3D model (GLB) from a single 2D image using one of several
 * free, publicly-hosted Hugging Face Spaces. Each Space has its own API
 * shape - the endpoint names and parameters below were confirmed by
 * reading each Space's own source file directly (not guessed):
 *   - tencent/Hunyuan3D-2 -> gradio_app.py (huggingface.co/spaces/tencent/Hunyuan3D-2)
 *   - stabilityai/TripoSR -> app.py (huggingface.co/spaces/stabilityai/TripoSR)
 *
 * NOTE: these are research demos, not stable public APIs. The Space
 * owners can change their code at any time, which would require updating
 * the calls below to match. If generation starts failing, re-check the
 * relevant Space's source file for its current function signature.
 */
export class ImageTo3DGenerator {
  private provider: ImageTo3DProvider = 'hunyuan3d2'
  private hf_token: string | undefined

  private on_progress: (status: string) => void = () => {}

  public set_progress_callback (callback: (status: string) => void): void {
    this.on_progress = callback
  }

  public set_provider (provider: ImageTo3DProvider): void {
    this.provider = provider
  }

  /**
   * Optional Hugging Face access token (from huggingface.co/settings/tokens).
   * Authenticated requests get a much larger free ZeroGPU quota than
   * anonymous ones.
   */
  public set_hf_token (token: string | undefined): void {
    this.hf_token = (token !== undefined && token.trim().length > 0) ? token.trim() : undefined
  }

  /**
   * Sends an image file to the selected provider and resolves with a
   * blob: URL pointing to the generated GLB file, ready to be passed
   * straight into StepLoadModel.load_model_file(url, 'glb').
   */
  public async generate_from_image (image_file: File): Promise<string> {
    if (this.provider === 'triposr') {
      return await this.generate_with_triposr(image_file)
    }

    return await this.generate_with_hunyuan3d2(image_file)
  }

  // ============================================================
  // tencent/Hunyuan3D-2
  // ============================================================

  private async generate_with_hunyuan3d2 (image_file: File): Promise<string> {
    const space_id = 'tencent/Hunyuan3D-2'
    const primary_api_name = '/generation_all' // textured, ~90s GPU budget
    const fallback_api_name = '/shape_generation' // untextured, ~40s GPU budget, more reliable

    this.on_progress('Connecting to generation service…')
    const client = await Client.connect(space_id, this.connect_options())

    try {
      return await this.run_hunyuan3d2_call(client, image_file, primary_api_name, fallback_api_name)
    } catch (error: unknown) {
      const message = this.describe_unknown_error(error)

      this.on_progress(
        `Textured generation failed (${message.slice(0, 80)}) - falling back to the faster untextured version…`
      )
      try {
        return await this.run_hunyuan3d2_call(client, image_file, fallback_api_name, fallback_api_name)
      } catch (fallback_error: unknown) {
        throw new Error(this.describe_unknown_error(fallback_error))
      }
    }
  }

  private async run_hunyuan3d2_call (
    client: Awaited<ReturnType<typeof Client.connect>>,
    image_file: File,
    api_name: string,
    fallback_api_name: string
  ): Promise<string> {
    this.on_progress(
      api_name === fallback_api_name
        ? 'Uploading image and generating 3D model… free queues can take 30s-2min'
        : 'Uploading image and generating textured 3D model… free queues can take 1-3min'
    )

    // positional args match shape_generation()'s / generation_all()'s signature in gradio_app.py:
    // (caption, image, mv_image_front, mv_image_back, mv_image_left, mv_image_right,
    //  steps, guidance_scale, seed, octree_resolution, check_box_rembg, num_chunks, randomize_seed)
    let result: Awaited<ReturnType<typeof client.predict>>
    try {
      result = await client.predict(api_name, [
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
    } catch (predict_error: unknown) {
      throw new Error(this.describe_unknown_error(predict_error))
    }

    return await this.resolve_glb_from_result(client, result.data)
  }

  // ============================================================
  // stabilityai/TripoSR
  // ============================================================

  private async generate_with_triposr (image_file: File): Promise<string> {
    const space_id = 'stabilityai/TripoSR'

    this.on_progress('Connecting to generation service…')
    const client = await Client.connect(space_id, this.connect_options())

    // TripoSR's UI chains 3 calls: check_input_image -> preprocess -> generate.
    // The check is just a null-check we already do client-side, so we skip
    // straight to preprocess -> generate, matching app.py's function signatures.
    this.on_progress('Removing background and preprocessing image…')
    let preprocess_result: Awaited<ReturnType<typeof client.predict>>
    try {
      preprocess_result = await client.predict('/preprocess', [
        image_file, // input_image
        true, // do_remove_background
        0.85 // foreground_ratio
      ])
    } catch (preprocess_error: unknown) {
      throw new Error(this.describe_unknown_error(preprocess_error))
    }

    const processed_image = preprocess_result.data[0]
    const processed_image_url = this.extract_file_url(processed_image, client)

    if (processed_image_url === null) {
      const raw_preview = JSON.stringify(preprocess_result.data, null, 2).slice(0, 500)
      throw new Error(
        'Could not find the preprocessed image URL in the response. Raw response ' +
        `(screenshot this and send it back): ${raw_preview}`
      )
    }

    this.on_progress('Generating 3D mesh… usually 10-30s')
    let generate_result: Awaited<ReturnType<typeof client.predict>>
    try {
      generate_result = await client.predict('/generate', [
        handle_file(processed_image_url), // must be wrapped, not passed as a raw object
        256 // mc_resolution (marching cubes resolution)
      ])
    } catch (generate_error: unknown) {
      throw new Error(this.describe_unknown_error(generate_error))
    }

    // generate() returns (obj_path, glb_path) - extractor already skips
    // non-.glb entries so the .obj path is ignored automatically
    return await this.resolve_glb_from_result(client, generate_result.data)
  }

  // ============================================================
  // shared helpers
  // ============================================================

  private connect_options (): { token: `hf_${string}` } | undefined {
    return this.hf_token !== undefined ? { token: this.hf_token as `hf_${string}` } : undefined
  }

  private async resolve_glb_from_result (
    client: Awaited<ReturnType<typeof Client.connect>>,
    data: unknown
  ): Promise<string> {
    const glb_path = this.extract_glb_url(data)

    if (glb_path === null) {
      const raw_preview = JSON.stringify(data, null, 2).slice(0, 500)
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
   * client.predict() can throw a plain Error, but Gradio's queue/status
   * errors (quota exceeded, GPU timeout, etc) often come through as a
   * plain object instead - which String(error) turns into a useless
   * "[object Object]". This pulls out whatever readable info exists.
   */
  private describe_unknown_error (error: unknown): string {
    if (error instanceof Error) {
      return error.message
    }

    if (typeof error === 'string') {
      return error
    }

    if (error !== null && typeof error === 'object') {
      const candidate = error as { message?: string, detail?: string, error?: string, stage?: string, type?: string }
      const readable = candidate.message ?? candidate.detail ?? candidate.error
      if (typeof readable === 'string') {
        const context = candidate.stage !== undefined ? ` (stage: ${candidate.stage})` : ''
        return `${readable}${context}`
      }

      try {
        return `Unrecognized error object: ${JSON.stringify(error, null, 2).slice(0, 500)}`
      } catch {
        return 'Unrecognized error object that could not be serialized.'
      }
    }

    return String(error)
  }

  /**
   * Pulls a usable file URL/path out of a single Gradio output item,
   * unwrapping gr.update() `.value` wrappers and handling both string
   * and object {url, path} shapes. Resolves relative paths against the
   * Space's own origin. Returns null if nothing file-like is found.
   */
  private extract_file_url (
    item: unknown,
    client: Awaited<ReturnType<typeof Client.connect>>
  ): string | null {
    let raw_url: string | null = null

    if (typeof item === 'string') {
      raw_url = item
    } else if (item !== null && typeof item === 'object') {
      const unwrapped = ('value' in item && item.value !== null && typeof item.value === 'object')
        ? item.value
        : item
      const candidate = unwrapped as { url?: string, path?: string }
      raw_url = candidate.url ?? candidate.path ?? null
    }

    if (raw_url === null) {
      return null
    }

    if (raw_url.startsWith('http')) {
      return raw_url
    }

    const client_config = (client as unknown as { config?: { root?: string } }).config
    const space_root = client_config?.root ?? ''
    return `${space_root}${raw_url}`
  }

  /**
   * Gradio endpoints return an array of outputs. File-type outputs
   * usually come back as objects with a `url` (or `path`) property
   * (sometimes wrapped in a gr.update() `.value`), but can also come
   * back as a plain string path/URL depending on the Space's Gradio
   * version. Some endpoints return multiple files (e.g. an untextured
   * AND textured mesh, or an .obj AND .glb) - this collects every GLB
   * found and prefers one whose filename suggests it's the textured
   * version, otherwise takes the first.
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
