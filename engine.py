import moderngl as mgl
import numpy as np
import glfw
from PIL import Image
import math




if not glfw.init():
    raise RuntimeError("GLFW init failed")

glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)

width = 1000
height = 700

one_width = 2/width
one_height = 2/height

window = glfw.create_window(width, height, "Графический движок", None, None)
glfw.make_context_current(window)
ctx = mgl.create_context(require=330)
# Настройка буфера глубины и текстуры
ctx.enable(mgl.DEPTH_TEST)
depth_tex = ctx.depth_texture((width, height))
fbo = ctx.framebuffer(depth_attachment=depth_tex)
# Загрузка шейдеров из файлов
load_shader_flag = True

max_lights = 32
vertex_shader = """
#version 330 core

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_color;

out vec3 color;

void main()
{
    gl_Position = vec4(in_position, 1.0);
    color = in_color;
}
"""

fragment_shader = """
#version 330 core

in vec3 color;
uniform float bright;
uniform vec3 lights[32];
uniform vec3 light_colors[32];
uniform float light_radii[32]; // Разные радиусы для каждого света
uniform vec2 resolution; 

out vec4 frag_color;

float distance3D(vec3 pointA, vec3 pointB) {
    return length(pointA - pointB);
}

void main()
{   
    vec3 pixel_coords = vec3(gl_FragCoord.x, resolution.y - gl_FragCoord.y, 0.0);
    vec3 final_light_color = vec3(0.0);
    
    for(int i = 0; i < 32; i++) {
        float dist = distance3D(pixel_coords, lights[i]);
        float influence = 1.0 - smoothstep(light_radii[i] * 0.5, light_radii[i], dist);
        final_light_color += light_colors[i] * influence;
    }
    
    final_light_color = min(final_light_color, vec3(1.0))-0.4;
    vec3 final_color = min(color * bright + final_light_color, vec3(1.0))+0.1;
    
    frag_color = vec4(final_color, 1.0);
}
"""

def mkn(kortege:tuple,number:int|float)->tuple:
    return tuple(x * number for x in kortege)

def mkk(kortege1:tuple,kortrge2:tuple)->tuple:
    return tuple(a * b for a, b in zip(kortege1, kortrge2))


light_positions = []
light_colors = []
light_radii = []
lights_count = 0

for i in range(max_lights):
    light_positions.append((0,0,0))
for i in range(max_lights):
    light_colors.append((0,0,0))
for i in range(max_lights):
    light_radii.append(0)

glfw.make_context_current(window)

def load_texture(path):
    img = Image.open(path)
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    img = img.convert('RGBA')
    texture = ctx.texture(img.size, 4, img.tobytes())
    texture.build_mipmaps()
    return texture




class graphical:
    def __init__(self):
        self.prev_size = (0,0)
        self.close_event = glfw.window_should_close(window)
    def update(self):
        self.close_event = glfw.window_should_close(window)
        glfw.poll_events()
        glfw.swap_buffers(window)
        self.current_size = glfw.get_window_size(window)

        if self.current_size != self.prev_size:
            print(f"Размер окна изменен с {self.prev_size} на {self.current_size}")
            self.all_update()
    def all_update(self):
        self.prev_size = self.current_size
        global one_width, one_height
        global width,height
        one_width = 2 / self.current_size[0]
        one_height = 2 / self.current_size[1]
        width,height = self.current_size[0], self.current_size[1]
        ctx.viewport = (0, 0, self.current_size[0], self.current_size[1])


class Color:
    def __init__(self,red_rgb:int,green_rgb:int,blue_rgb:int):
        one_rgb = 1/255
        if red_rgb > 255:
            red_rgb = 255
        if green_rgb > 255:
            red_rgb = 255
        if blue_rgb > 255:
            red_rgb = 255
        self.red = red_rgb*one_rgb
        self.green = green_rgb*one_rgb
        self.blue = blue_rgb*one_rgb
        self.one = [self.red,self.green,self.blue]
        self.two = [self.red, self.green, self.blue]
        self.three = [self.red, self.green, self.blue]
class Gradient:
    def __init__(self,colors:list):
        colors_converted = []
        for color in colors:
            red_rgb = color[0]
            green_rgb = color[1]
            blue_rgb = color[2]
            one_rgb = 1/255
            if red_rgb > 255:
                red_rgb = 255
            if green_rgb > 255:
                red_rgb = 255
            if blue_rgb > 255:
                red_rgb = 255
            red = red_rgb*one_rgb
            green = green_rgb*one_rgb
            blue = blue_rgb*one_rgb
            colors_converted.append([red,green,blue])
        self.one = colors_converted[0]
        self.two = colors_converted[1]
        self.three = colors_converted[2]

class drawing:
    def __init__(self):
        pass
    def image(self,texture,points:list,position = [0,0,0],bright =1.0):
        def convert_coords_x(x):
            x_converted = -1.0+(x*one_width)
            return x_converted
        def convert_coords_y(y):
            y_converted = 1.0 - (y * one_height)
            y_converted = -- y_converted
            return y_converted
        def convert_position_x(x):
            x_converted = x*one_width
            return x_converted
        def convert_position_y(y):
            y_converted = y * one_height
            y_converted = - y_converted
            return y_converted
        converted_position_x = convert_position_x(position[0])
        converted_position_y = convert_position_y(position[1])
        prog = ctx.program(
            vertex_shader="""
            #version 330
            in vec2 in_vert;
            in vec2 in_texcoord;
            out vec2 uv;
            uniform vec2 pos;
            uniform vec2 scale;
            void main() {
                uv = in_texcoord;
                gl_Position = vec4(pos + in_vert * scale, 0.0, 1.0);
            }
            """,
            fragment_shader="""
#version 330
in vec2 uv;
out vec4 frag_color;
uniform sampler2D tex;

uniform float bright;
uniform vec3 lights[32];
uniform vec3 light_colors[32];
uniform float light_radii[32]; // Разные радиусы для каждого света
uniform vec2 resolution; 

float distance3D(vec3 pointA, vec3 pointB) {
    return length(pointA - pointB);
}

void main() {
        vec3 pixel_coords = vec3(gl_FragCoord.x, resolution.y - gl_FragCoord.y, 0.0);
    vec3 final_light_color = vec3(0.0);
    
    for(int i = 0; i < 32; i++) {
        float dist = distance3D(pixel_coords, lights[i]);
        float influence = 1.0 - smoothstep(light_radii[i] * 0.5, light_radii[i], dist);
        final_light_color += light_colors[i] * influence;
    }
    
    final_light_color = min(final_light_color, vec3(1.0))-0.4;
    
    vec4 texture_color = texture(tex, vec2(uv.x, 1.0 - uv.y))+0.4;
    
    frag_color = texture_color*bright + vec4(final_light_color,0.0)*bright;
}
            """
        )
        quad = ctx.buffer(np.array([
            # position   texcoords
            convert_coords_x(points[3][0])+converted_position_x,convert_coords_y(points[3][1])+converted_position_y, 0, 1,
            convert_coords_x(points[2][0])+converted_position_x,convert_coords_y(points[2][1])+converted_position_y, 1, 1,
            convert_coords_x(points[0][0])+converted_position_x,convert_coords_y(points[0][1])+converted_position_y, 0, 0,
            convert_coords_x(points[1][0])+converted_position_x,convert_coords_y(points[1][1])+converted_position_y, 1, 0
        ], dtype='f4'))

        vao = ctx.vertex_array(prog, [
            (quad, '2f 2f', 'in_vert', 'in_texcoord')
        ])

        # Критически важные настройки прозрачности
        ctx.enable(mgl.BLEND)
        ctx.blend_equation = mgl.FUNC_ADD
        ctx.blend_func = (
            mgl.SRC_ALPHA,  # Источник: альфа-канал
            mgl.ONE_MINUS_SRC_ALPHA  # Назначение: 1 - альфа
        )

        # Загрузка и отрисовка

        prog["bright"].value = bright

        prog[f'lights'].value = light_positions
        prog[f'light_colors'].value = light_colors
        prog[f'light_radii'].value = light_radii
        prog['pos'].value = (0, 0)
        prog['scale'].value = (1.0, 1.0)
        prog['resolution'].value = (width, height)
        texture.use(0)

        vao.render(mode = mgl.TRIANGLE_STRIP)
    def triangle(self,points:list,color,position = [0,0,0],bright = 1.0):
        if len(points) == 3:
            pass
        else:
            print("Ошибка TRIANGLE более или менее 3 точек")
            return
        def convert_coords_x(x):
            x_converted = -1.0+(x*one_width)
            return x_converted
        def convert_coords_y(y):
            y_converted = 1.0 - (y * one_height)
            y_converted = -- y_converted
            return y_converted
        def convert_position_x(x):
            x_converted = x*one_width
            return x_converted
        def convert_position_y(y):
            y_converted = y * one_height
            y_converted = - y_converted
            return y_converted
        if position[2] > 1.0:
            position[2] = 1.0
        if position[2] < -1.0:
            position[2] = -1.0
        converted_position_x = convert_position_x(position[0])
        converted_position_y = convert_position_y(position[1])
        converted_position_z = position[2]
        # Создание шейдерной программы
        vertices = np.array([
            convert_coords_x(points[0][0]) + converted_position_x,
            convert_coords_y(points[0][1]) + converted_position_y,
            0.0 + converted_position_z, color.one[0], color.one[1], color.one[2],
            convert_coords_x(points[1][0]) + converted_position_x,
            convert_coords_y(points[1][1]) + converted_position_y,
            0.0 + converted_position_z, color.two[0], color.two[1], color.two[2],
            convert_coords_x(points[2][0]) + converted_position_x,
            convert_coords_y(points[2][1]) + converted_position_y,
            0.0 + converted_position_z, color.three[0], color.three[1], color.three[2],
        ], dtype='f4')
        vbo = ctx.buffer(vertices)
        prog = ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader
        )
        prog["bright"].value = bright

        prog[f'lights'].value = light_positions
        prog[f'light_colors'].value = light_colors
        prog[f'light_radii'].value = light_radii
        prog['resolution'].value = (width, height)

        vao = ctx.vertex_array(
            prog,
            [(vbo, '3f 3f', 'in_position', 'in_color')]
        )

        vao.render(mode = mgl.TRIANGLES)
    def line(self,start_point:list,end_point:list,color,bright = 1.0):
        def convert_coords_x(x):
            x_converted = -1.0+(x*one_width)
            return x_converted
        def convert_coords_y(y):
            y_converted = 1.0 - (y * one_height)
            y_converted = -- y_converted
            return y_converted
        def convert_coords_z(z):
            if z > 1.0:
                z = 1.0
            if z < -1.0:
                z = -1.0
            return z
        prog = ctx.program(
            vertex_shader='''
            #version 330
            in vec2 in_pos;
            void main() {
                gl_Position = vec4(in_pos, 0.0, 1.0);
            }
            ''',
            fragment_shader='''
            #version 330
            uniform vec3 color;
            uniform float bright;
            uniform vec3 lights[32];
            uniform vec3 light_colors[32];
            uniform float light_radii[32]; // Разные радиусы для каждого света
            uniform vec2 resolution; 
            
            out vec4 frag_color;
            
            float distance3D(vec3 pointA, vec3 pointB) {
                return length(pointA - pointB);
            }
            
            void main()
            {   
                vec3 pixel_coords = vec3(gl_FragCoord.x, resolution.y - gl_FragCoord.y, 0.0);
                vec3 final_light_color = vec3(0.0);
                
                for(int i = 0; i < 32; i++) {
                    float dist = distance3D(pixel_coords, lights[i]);
                    float influence = 1.0 - smoothstep(light_radii[i] * 0.5, light_radii[i], dist);
                    final_light_color += light_colors[i] * influence;
                }
                
                final_light_color = min(final_light_color, vec3(1.0))-0.4;
                frag_color = vec4(color*bright+final_light_color, 1.0);
            }
            '''
        )
        prog["bright"].value = bright

        #prog[f'lights'].value = light_positions
        #prog[f'light_colors'].value = light_colors
        #prog[f'light_radii'].value = light_radii
        # Вершины линии (2 точки)
        line_vertices = np.array([
            convert_coords_x(start_point[0]), convert_coords_y(start_point[1]),convert_coords_z(start_point[2]),
            convert_coords_x(end_point[0]), convert_coords_y(end_point[1]),convert_coords_z(end_point[2])
        ], dtype='f4')

        # Создание буферов
        vbo = ctx.buffer(line_vertices)
        vao = ctx.vertex_array(prog, [(vbo, '3f', 'in_pos')])
        prog['color'].value = (color.one[0],color.one[1],color.one[2])

        vao.render(mode = mgl.LINES)
    def quad(self,points:list,color,position = [0,0,0],bright = 1.0):
        if len(points) == 4:
            first_triangle_points = [points[0],points[1],points[3]]
            second_triangle_points = [points[2],points[1],points[3]]
            self.triangle(first_triangle_points,color,position,bright=bright)
            self.triangle(second_triangle_points,color,position,bright=bright)
        else:
            print("Ошибка QUAD более или менее 4 точек")
    def fill(self,color):
        ctx.clear(color.one[0], color.one[1], color.one[2], depth=1.0)
        fbo.use()
        ctx.clear(depth=1.0)
        ctx.screen.use()
    def circle(self,center:list,radius:int,color,segments=32,bright=1,position = [0,0,0]):
        points = [center]
        for i in range(segments+1):
            angle = i *2 *math.pi/segments
            x = math.cos(angle)*radius
            y = math.sin(angle)*radius
            points.append([x+center[0],y+center[1],0.0])
        for i in range(len(points)-2):
            second_point = i + 1
            third_point = i + 2
            if third_point == len(points):
                third_point = 1
            self.triangle([points[0],points[second_point],points[third_point]],color,position=position,bright = bright)
    def polygon(self,points:list,color,bright=1.0,position = [0,0,0]):
        for i in range(len(points)-1):
            second_point = i + 1
            third_point = i + 2
            if third_point == len(points):
                third_point = 0
            self.triangle([points[0],points[second_point],points[third_point]],color,position=position,bright = bright)
        for i in range(len(points)):
            drow = drawing()
            end_pos = i + 1
            if end_pos == len(points):
                end_pos = 0
            drow.line(points[i],points[end_pos],Color(255,255,255))

class Rect:
    def __init__(self,x,y,width,height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.draw_class = drawing()

        self.min_x = self.x
        self.min_y = self.y
        self.max_x = self.x + self.width
        self.max_y = self.y + self.height
    def __str__(self):
        return "Rect"
    def draw(self,color,position = [0,0,0],bright = 1):
        self.draw_class.quad([[self.x,self.y,0],[self.x,self.max_y,0],[self.max_x, self.max_y,0],[self.max_x,self.y,0]],color,position,bright = bright)
    def coliderrect(self,another_rect):
        flag = False
        if another_rect.min_x <= self.max_x and another_rect.min_x >= self.min_x-self.width:
            if another_rect.min_y <= self.max_y and another_rect.min_y >= self.min_y-self.height:
                flag = True
        return flag
    def update(self,x,y,width,height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

        self.min_x = self.x
        self.min_y = self.y
        self.max_x = self.x + self.width
        self.max_y = self.y + self.height
    def update_x(self,x):
        self.x = x
        self.min_x = self.x
        self.max_x = self.x + self.width
    def update_y(self,y):
        self.y = y
        self.min_y = self.y
        self.max_y = self.y + self.height
    def update_width(self,width):
        self.width = width
        self.max_x = self.x + self.width
    def update_height(self,height):
        self.height = height
        self.max_y = self.y + self.height
class Keyboard:
    def __init__(self):
        pass
    def if_key(self,key:str):
        return glfw.get_key(window,key)

class Light:
    def __init__(self,x,y,intensive=1.0,color=(0,0,0),radius = 100):
        self.x = x
        self.y = y
        self.color = mkn(color,intensive)

        self.radius = radius
        global light_colors, light_radii, light_positions,lights_count
        self.i = lights_count
        light_positions[self.i] = (self.x,self.y,0)
        light_colors[self.i] = self.color
        light_radii[self.i] = self.radius
        lights_count += 1
    def destroy(self):
        global light_colors, light_radii, light_positions, lights_count
        lights_count -= 1
        light_positions.pop(self.i)
        light_colors.pop(self.i)
        light_radii.pop(self.i)
        light_positions.append((0,0,0))
        light_colors.append((0,0,0))
        light_radii.append(0.0)


engine = graphical()
draw = drawing()
keys= Keyboard()

background = Rect(0,0,1000,1000)

q = Rect(0,0,100,100)
q2 = Rect(250,250,100,100)


l = Light(200,200,color = (2,1,0),radius=100,intensive=1.0)


texture = load_texture("texture.jpg")

run = True

POS_X = 0
POS_Y = 0

while run:
    if engine.close_event:
        run = False
    engine.update()
    if keys.if_key(glfw.KEY_D) == glfw.PRESS:
        POS_X += 2
        q.update_x(POS_X)
        print(q.coliderrect(q2))
    if keys.if_key(glfw.KEY_A) == glfw.PRESS:
        POS_X -= 2
        q.update_x(POS_X)
        print(q.coliderrect(q2))
    if keys.if_key(glfw.KEY_W) == glfw.PRESS:
        POS_Y -= 2
        q.update_y(POS_Y)
        print(q.coliderrect(q2))
    if keys.if_key(glfw.KEY_S) == glfw.PRESS:
        POS_Y += 2
        q.update_y(POS_Y)
        print(q.coliderrect(q2))
    draw.fill(Color(30,30,30))

    q.draw(Gradient([[255,0,0],[0,0,255],[0,255,0]]),bright = 2.0)
    draw.line([0,0,0],[500,500,0],Color(255,0,255))
    draw.circle([500,500,0],50,Color(255,0,255))
    draw.polygon([[200,200,0],[250,200,0],[265,245,0],[156,250,0]],Color(255,20,255))
    draw.image(texture, [[100, 100, 0], [100, 200, 0], [200, 200, 0], [200, 100, 0]],bright=1.0)
    background.draw(Color(50, 50, 50),bright=1.0)


glfw.terminate()