import { useEffect, useState } from 'react';
import { loadSignLibrary } from '../lib/signLibrary';
import type { SignLibrary } from '../lib/reverse';
export function useSignLibrary() {
  const [library, setLibrary] = useState<SignLibrary>({});
  const [error, setError] = useState('');
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    let cancelled = false;
    setError('');
    loadSignLibrary().then(data => { if (!cancelled) setLibrary(data); }).catch(e => {
      if (!cancelled) setError(e instanceof Error ? e.message : String(e));
    });
    return () => { cancelled = true; };
  }, [attempt]);
  return { library, error, loading: !error && !Object.keys(library).length, retry: () => setAttempt(n => n + 1) };
}
