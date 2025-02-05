/**
 * Loads weights from a URL.
 * @param url The url to load the weights from.
 */
export default async function loadWeights(url: string): Promise<Float32Array> {
	const res = await fetch(url);
	const buffer = await res.arrayBuffer();
	return new Float32Array(buffer);
}
