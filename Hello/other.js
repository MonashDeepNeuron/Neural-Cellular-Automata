export function performRenderPass(
    renderPass, pipeline, vertexBuffer, bindGroup
) {
    renderPass.setPipeline(pipeline);
    renderPass.setVertexBuffer(0, vertexBuffer);
    renderPass.setBindGroup(0, bindGroup);
    renderPass.draw(3, 1);
    renderPass.end();
}



// Canvas, Device, 
// Texture, Buffer, Bind Groups, Shaders, (this part might be different between models)
// Pipeline, Encoder, Pass