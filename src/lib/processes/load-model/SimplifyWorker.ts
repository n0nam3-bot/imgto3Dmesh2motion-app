import { MeshoptSimplifier } from 'meshoptimizer'

/**
 * Runs mesh simplification (meshoptimizer) in a worker instead of on the
 * main thread. Same reasoning as UvUnwrapWorker.ts: this is synchronous,
 * CPU-bound WASM work that can freeze the page on dense (AI-generated)
 * meshes with no way to recover. Confirmed to freeze in practice on a
 * ~1M-triangle mesh even though meshoptimizer is normally very fast -
 * "normally fast" still isn't instant at that scale, and any nonzero
 * synchronous delay on the main thread blocks the whole page.
 */

export interface SimplifyRequest {
  positions: Float32Array
  indices: Uint32Array
  target_index_count: number
}

export interface SimplifySuccessResponse {
  status: 'success'
  indices: Uint32Array
}

export interface SimplifyErrorResponse {
  status: 'error'
  message: string
}

export type SimplifyResponse = SimplifySuccessResponse | SimplifyErrorResponse

self.onmessage = (event: MessageEvent<SimplifyRequest>) => {
  void run_simplify(event.data)
}

async function run_simplify (request: SimplifyRequest): Promise<void> {
  try {
    await MeshoptSimplifier.ready

    const [new_indices] = MeshoptSimplifier.simplify(
      request.indices,
      request.positions,
      3,
      request.target_index_count,
      1.0
    )

    // copy defensively before transferring - same lesson learned from the
    // UV worker: don't hand back a library-owned buffer
    const response: SimplifySuccessResponse = {
      status: 'success',
      indices: Uint32Array.from(new_indices)
    }

    // @ts-expect-error - self here is a DedicatedWorkerGlobalScope
    self.postMessage(response, [response.indices.buffer])
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error)
    const response: SimplifyErrorResponse = { status: 'error', message }
    self.postMessage(response)
  }
}
