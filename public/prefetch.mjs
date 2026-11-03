// Finish every request (including cancellation) before the caller starts training.
export async function prefetchAll(items, worker, {signal, concurrency=16, onProgress=()=>{}}={}) {
  if (!Number.isInteger(concurrency) || concurrency < 1) throw Error('取得並列数を確認してください');
  signal?.throwIfAborted();
  const controller = new AbortController();
  const combined = signal ? AbortSignal.any([signal, controller.signal]) : controller.signal;
  let next = 0, done = 0, failure;
  onProgress(0, items.length);
  const jobs = Array.from({length: Math.min(concurrency, items.length)}, async () => {
    try {
      while (next < items.length) {
        combined.throwIfAborted();
        const item = items[next++];
        await worker(item, combined);
        combined.throwIfAborted();
        onProgress(++done, items.length);
      }
    } catch (error) {
      failure ??= error;
      controller.abort(error);
      throw error;
    }
  });
  await Promise.allSettled(jobs);
  if (failure) throw failure;
  combined.throwIfAborted();
  return done;
}
