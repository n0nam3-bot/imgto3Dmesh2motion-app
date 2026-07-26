import createXAtlas from 'xatlas-wasm'

/**
 * Runs xatlas UV unwrapping in a worker instead of on the main thread.
 *
 * This exists because xatlas's unwrap computation is synchronous, CPU-bound
 * WASM work - running it directly on the main thread blocks the entire
 * page (confirmed: it triggered the browser's "page unresponsive" warning
 * on a real AI-generated mesh, with no way to recover except closing the
 * tab). A blocked main thread also can't run a JS timeout/Promise.race to
 * rescue itself, since the event loop itself is frozen. Moving the work
 * here keeps the page responsive throughout, and lets the caller
 * worker.terminate() this if it takes too long.
 */

export interface UvUnwrapRequest {
  positions: Float32Array
  indices: Uint32Array | undefined
  normals: Float32Array | undefined
}

export interface UvUnwrapSuccessResponse {
  status: 'success'
  positions: Float32Array
  uvs: Float32Array
  normals: Float32Array | undefined
  indices: Uint32Array
}

export interface UvUnwrapErrorResponse {
  status: 'error'
  message: string
}

export type UvUnwrapResponse = UvUnwrapSuccessResponse | UvUnwrapErrorResponse

self.onmessage = (event: MessageEvent<UvUnwrapRequest>) => {
  void run_unwrap(event.data)
}

async function run_unwrap (request: UvUnwrapRequest): Promise<void> {
  try {
    const xatlas = await createXAtlas()
    const atlas = xatlas.createAtlas()

    try {
      const add_result = atlas.addMesh({
        positions: request.positions,
        indices: request.indices
      })

      if (add_result !== 0) {
        throw new Error(`xatlas addMesh failed: ${xatlas.addMeshErrorString(add_result)}`)
      }

      atlas.generate()

      const output_mesh = atlas.getMesh(0)
      const output_vertex_count = output_mesh.vertexCount

      const has_normals = request.normals !== undefined
      const new_positions = new Float32Array(output_vertex_count * 3)
      const new_uvs = new Float32Array(output_vertex_count * 2)
      const new_normals = has_normals ? new Float32Array(output_vertex_count * 3) : undefined

      for (let i = 0; i < output_vertex_count; i++) {
        const vertex = output_mesh.vertices[i]
        const source_index = vertex.xref

        new_positions[i * 3] = request.positions[source_index * 3]
        new_positions[i * 3 + 1] = request.positions[source_index * 3 + 1]
        new_positions[i * 3 + 2] = request.positions[source_index * 3 + 2]

        new_uvs[i * 2] = vertex.uv[0] / atlas.width
        new_uvs[i * 2 + 1] = vertex.uv[1] / atlas.height

        if (new_normals !== undefined && request.normals !== undefined) {
          new_normals[i * 3] = request.normals[source_index * 3]
          new_normals[i * 3 + 1] = request.normals[source_index * 3 + 1]
          new_normals[i * 3 + 2] = request.normals[source_index * 3 + 2]
        }
      }

      const response: UvUnwrapSuccessResponse = {
        status: 'success',
        positions: new_positions,
        uvs: new_uvs,
        normals: new_normals,
        // IMPORTANT: output_mesh.indices is very likely a live view into the
        // WASM module's own linear memory, not an independent buffer. It
        // must be COPIED before being transferred/used after atlas.destroy() -
        // transferring the original would hand over (and detach) a chunk of
        // the WASM heap itself, corrupting subsequent WASM state and
        // producing garbage geometry (this was confirmed to actually happen).
        indices: Uint32Array.from(output_mesh.indices)
      }

      const transfer_list: ArrayBuffer[] = [
        new_positions.buffer,
        new_uvs.buffer,
        response.indices.buffer
      ]
      if (new_normals !== undefined) {
        transfer_list.push(new_normals.buffer)
      }

      // @ts-expect-error - self here is a DedicatedWorkerGlobalScope, postMessage takes a transfer list
      self.postMessage(response, transfer_list)
    } finally {
      atlas.destroy()
    }
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error)
    const response: UvUnwrapErrorResponse = { status: 'error', message }
    self.postMessage(response)
  }
}
