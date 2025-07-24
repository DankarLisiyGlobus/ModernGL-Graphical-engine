#version 330 core

in vec3 color;
out vec4 frag_color;

// Простая псевдослучайная функция
float rand(vec2 co) {
    return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
}

void main()
{
    // Генерируем случайное отклонение от -0.1 до 0.1
    float randomOffset = rand(gl_FragCoord.xy) * 0.02 - 0.1;

    // Применяем к каждому цветовому каналу
    vec3 noisyColor = color + vec3(randomOffset);

    // Ограничиваем значения цвета
    noisyColor = clamp(noisyColor, 0.0, 1.0);

    frag_color = vec4(noisyColor, 1.0);
}