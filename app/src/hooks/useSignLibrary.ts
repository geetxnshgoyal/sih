import { useCallback, useEffect, useState } from 'react';
import { loadSignCatalog, loadSignClip } from '../lib/signLibrary';
import type { SignClip, SignLibrary } from '../lib/reverse';

export function useSignLibrary() {
  const [library, setLibrary] = useState<SignLibrary>({});
  const [error, setError] = useState('');
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    let cancelled = false;
    setError('');
    loadSignCatalog().then(names => {
      if (!cancelled) setLibrary(Object.fromEntries(names.map(name => [name, {version: 2, fps: 14, body: []}])));
    }).catch(reason => {
      if (!cancelled) setError(reason instanceof Error ? reason.message : String(reason));
    });
    return () => { cancelled = true; };
  }, [attempt]);

  const getClip = useCallback(async (gloss: string): Promise<SignClip | null> => {
    try {
      const frames = await loadSignClip(gloss);
      if (frames) setLibrary(previous => ({ ...previous, [gloss]: frames }));
      return frames;
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
      return null;
    }
  }, []);

  return {
    library, getClip, error,
    loading: !error && !Object.keys(library).length,
    retry: () => setAttempt(value => value + 1),
  };
}
