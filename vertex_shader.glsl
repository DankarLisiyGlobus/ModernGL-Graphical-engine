#version 330 core

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_color;

uniform mat3 transform;
out vec3 color;

void main()
{
    vec2 pos = (transform * vec3(in_position.xy, 1.0)).xy;
    gl_Position = vec4(pos, in_position.z, 1.0);
    color = in_color;
}