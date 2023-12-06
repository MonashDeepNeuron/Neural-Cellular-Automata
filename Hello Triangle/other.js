export function performRenderPass(
    renderPass, pipeline, vertexBuffer, bindGroup
) {
    renderPass.setPipeline(pipeline);
    renderPass.setVertexBuffer(0, vertexBuffer);
    renderPass.setBindGroup(0, bindGroup);
    renderPass.draw(3, 1);
    renderPass.end();
}