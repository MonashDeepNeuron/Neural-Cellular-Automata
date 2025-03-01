/**
 * Loads weights from a URL.
 * @param url The url to load the weights from.
 */
export default async function loadWeights(url: string): Promise<Float32Array> {
	const res = await fetch(url);
	if (!res.ok) throw new Error(`${res.status}: ${res.statusText}`);
	const buffer = await res.arrayBuffer();
	return new Float32Array(buffer);
}
