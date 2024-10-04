export { loadWeights };

async function loadWeights(url) {
    try {
        const response = await fetch(url);
        const buffer = await response.arrayBuffer();
        const float32Array = new Float32Array(buffer);
        return float32Array;
    } catch (error) {
        // console.error('Error fetching the file:', error);
        throw error;
    }
}
