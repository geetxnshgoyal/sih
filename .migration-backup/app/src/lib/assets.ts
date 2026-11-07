/** Asset errors must not be mistaken for an empty vocabulary or an HTML SPA fallback. */
export async function fetchJson<T>(url: string): Promise<T> {
  const response = await fetch(url, { cache: 'no-cache', signal: AbortSignal.timeout(30000) });
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  try { return await response.json() as T; }
  catch { throw new Error(`${url}: expected JSON; check that this asset was exported`); }
}
