#version 450

layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// Storage image we write into
layout (binding = 0, rgba32f) uniform writeonly image2D outImage;

// Push constant: simple time value for animation
layout (push_constant) uniform PushData {
    float time;
} uPush;

void main() {
    ivec2 size  = imageSize(outImage);
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);

    if (coord.x >= size.x || coord.y >= size.y)
        return;

    vec2 uv = (vec2(coord) + 0.5) / vec2(size);

    // Moving stripe pattern based on time
    float stripe = 0.5 + 0.5 * sin(20.0 * uv.x + uPush.time * 5.0);

    vec4 color = vec4(uv.x, stripe, uv.y, 1.0);
    imageStore(outImage, coord, color);
}
