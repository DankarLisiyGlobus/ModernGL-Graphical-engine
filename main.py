import moderngl as mgl
import numpy as np
import glfw
from PIL import Image
import math
import time
import sys


def load_texture(path):
    img = Image.open(path)
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    img = img.convert('RGBA')
    texture = ctx.texture(img.size, 4, img.tobytes())
    texture.build_mipmaps()
    return texture
def get_edited_keys(dict1, dict2):
    # Ключи только в dict1
    only_dict1 = set(dict1) - set(dict2)

    # Ключи в обоих, но с разными значениями
    common_keys_diff = {
        key for key in dict1.keys() & dict2.keys()
        if dict1[key] != dict2[key]
    }

    # Объединяем и преобразуем в список
    return list(only_dict1 | common_keys_diff)
def load_shader(file_path):
    """Загружает шейдер из файла"""
    try:
        with open(file_path, 'r',encoding="utf-8") as file:
            return file.read()
    except Exception as e:
        print(f"Error loading shader {file_path}: {e}")
        raise

# Инициализация GLFW
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


vertex_shader = load_shader("vertex_shader.glsl")
fragment_shader = load_shader("fragment_shader.glsl")


glfw.make_context_current(window)



class promt:
    def __init__(self,type:str,points:list,position:list,color,rotation:float,path_image:str = ""):
        self.type = type
        self.points = points
        self.position = position
        self.rotation = rotation
        self.color = color
        self.path_image = path_image
class graphical:
    def __init__(self):
        self.prev_size = (0,0)
    def update(self):
        glfw.poll_events()
        glfw.swap_buffers(window)
        self.current_size = glfw.get_window_size(window)

        if self.current_size != self.prev_size:
            print(f"Size changed from {self.prev_size} to {self.current_size}")
            self.all_update()
    def all_update(self):
        self.prev_size = self.current_size
        global one_width, one_height
        one_width = 2 / self.current_size[0]
        one_height = 2 / self.current_size[1]
        for object in objects:
            if objects[object].type == "triangle":
                draw.triangle(object, objects[object].points, objects[object].color, objects[object].position,
                              rotation=objects[object].rotation)
            elif objects[object].type == "line":
                draw.line(object, objects[object].points[0], objects[object].points[1], objects[object].color)
            elif objects[object].type == "image":
                draw.image(object, objects[object].path_image, objects[object].points, objects[object].position)
            else:
                print(f"Error draw object {object}: unknown type({objects[object].type})")
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
    def image(self,name,path,points:list,position:list):
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
            void main() {
                vec4 tex_color = texture(tex, uv);
                //frag_color = tex_color;
                // Альтернативный вариант с premultiplied alpha:
                frag_color = texture(tex, vec2(uv.x, 1.0 - uv.y));
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
        texture = load_texture(path)
        prog['pos'].value = (0, 0)
        prog['scale'].value = (1.0, 1.0)
        texture.use(0)

        objects[name] = promt("image", points, position, None, 0, path_image=path)
        objects_vao[name] = vao

        return vao
    def triangle(self,name,points:list,color,position,rotation:float = 0,texture = None):
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
        if texture != None:
            # Создание шейдерной программы
            try:
                vertices = np.array([
                    convert_coords_x(points[0][0]) + converted_position_x,convert_coords_y(points[0][1]) + converted_position_y,  0, 1,

                    convert_coords_x(points[1][0]) + converted_position_x,convert_coords_y(points[1][1]) + converted_position_y,  1, 1,

                    convert_coords_x(points[2][0]) + converted_position_x,convert_coords_y(points[2][1]) + converted_position_y,  0, 0,
                ], dtype='f4')
                vbo = ctx.buffer(vertices)
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
                    void main() {
                        vec4 tex_color = texture(tex, uv);
                        //frag_color = tex_color;
                        // Альтернативный вариант с premultiplied alpha:
                        frag_color = texture(tex, vec2(uv.x, 1.0 - uv.y));
                    }
                    """
                )
                vao = ctx.vertex_array(prog, [
                    (vbo, '3f 2f', 'in_vert','in_texcoord')
                ])
                # Критически важные настройки прозрачности
                ctx.enable(mgl.BLEND)
                ctx.blend_equation = mgl.FUNC_ADD
                ctx.blend_func = (
                    mgl.SRC_ALPHA,  # Источник: альфа-канал
                    mgl.ONE_MINUS_SRC_ALPHA  # Назначение: 1 - альфа
                )

                # Загрузка и отрисовка
                texture = load_texture(texture)
                prog['pos'].value = (0, 0)
                prog['scale'].value = (1.0, 1.0)
                texture.use(0)
            except Exception as e:
                print("Shader program error:", e)
                glfw.terminate()
                exit()
        else:
            try:
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
                vao = ctx.vertex_array(
                    prog,
                    [(vbo, '3f 3f', 'in_position', 'in_color')]
                )
            except Exception as e:
                print("Shader program error:", e)
                glfw.terminate()
                exit()

        objects[name] = promt("triangle",points,position,color,rotation)
        objects_vao[name] = vao

        if texture == None:
            transform = identity()
            transform = rotate(rotation)

            prog['transform'].write(transform.tobytes())
        return  vao
    def line(self,name,start_point:list,end_point:list,color):
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
            out vec4 frag_color;
            uniform vec3 color;
            void main() {
                frag_color = vec4(color, 1.0);
            }
            '''
        )

        # Вершины линии (2 точки)
        line_vertices = np.array([
            convert_coords_x(start_point[0]), convert_coords_y(start_point[1]),convert_coords_z(start_point[2]),
            convert_coords_x(end_point[0]), convert_coords_y(end_point[1]),convert_coords_z(end_point[2])
        ], dtype='f4')

        # Создание буферов
        vbo = ctx.buffer(line_vertices)
        vao = ctx.vertex_array(prog, [(vbo, '3f', 'in_pos')])
        prog['color'].value = (color.one[0],color.one[1],color.one[2])

        objects[name] = promt("line", [start_point,end_point], [0,0,0], color, 0)
        objects_vao[name] = vao
        return vao
    def text(self,name:str,text:str,position:list):
        pass
    def quad(self,name,points:list,color,position,rotation = 0,texture = None):
        first_triangle_points = [points[0],points[1],points[3]]
        second_triangle_points = [points[2],points[1],points[3]]
        self.triangle(f"{name}-in-triangle-1",first_triangle_points,color,position,rotation,texture)
        self.triangle(f"{name}-in-triangle-2", second_triangle_points,color,position,rotation,texture)
    def fill(self,color):
        ctx.clear(color.one[0], color.one[1], color.one[2], depth=1.0)
        fbo.use()
        ctx.clear(depth=1.0)
        ctx.screen.use()
class Transform:
    def __init__(self):
        pass
    def get_position(self,name):
        if name in objects:
            return objects[name].position
        else:
            print(f"Error get position for object {name}: unkown name {name}")
    def rotate(self,name,angle):
        if name in objects:
            objects[name].rotation = objects[name].rotation+angle
            draw.triangle(name, objects[name].points,objects[name].color,objects[name].position,rotation = objects[name].rotation)
        else:
            print(f"Error rotation object {name}: unkown name {name}")
    def set_position(self,name,position):
        if name in objects:
            objects[name].position = position
            draw.triangle(name, objects[name].points,objects[name].color,objects[name].position)
        else:
            print(f"Error set position for object {name}: unkown name {name}")
class keys:
    def __init__(self):
        pass
    def if_pressed(self,key):
        try:
            key = glfw.get_key(window,key)
        except:
            print(f"Error key name {key}")
        return key
# Функции для работы с матрицами 2D
def identity():
    return np.identity(3, dtype='f4')

def translate(x, y):
    m = identity()
    m[0, 2] = x
    m[1, 2] = y
    return m

def rotate(angle):
    s = math.sin(angle)
    c = math.cos(angle)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ], dtype='f4')

def scale(sx, sy):
    m = identity()
    m[0, 0] = sx
    m[1, 1] = sy
    return m

objects = {}
objects_last = {}
objects_vao = {}
transform = Transform()
draw = drawing()


class game:
    def setup(self):
        pass
    def update(self):
        pass



gp = graphical()
# Главный цикл

fps_counter = 0
fps_last_time = time.time()
fps = 0
glfw.set_time(0.0)
last_time = 0.0
engine = glfw
Keys = keys()


TARGET_FPS = 60
one_cadr_time = 1/TARGET_FPS
frame_time_t = 1.0 / TARGET_FPS
last_time_t = time.time()

gam = game()
setup = gam.setup
update = gam.update
#В setup() пишешь код игры ,выполняющийся один раз при запуске
setup()

i = 0
while not engine.window_should_close(window):
    i +=0.1
    gp.update()
    draw.fill(Color(0, 0, 0))
    update_vaos = get_edited_keys(objects,objects_last)
    for object in update_vaos:
        if objects[object].type == "triangle":
            draw.triangle(object,objects[object].points,objects[object].color,objects[object].position,rotation = objects[object].rotation)
        elif objects[object].type == "line":
            draw.line(object,objects[object].points[0],objects[object].points[1],objects[object].color)
        elif objects[object].type == "image":
            draw.image(object,objects[object].path_image,objects[object].points,objects[object].position)
        else:
            print(f"Error draw object {object}: unknown type({objects[object].type})")
    for object_vao in objects_vao.keys():
        if objects[object_vao].type == "triangle":
            objects_vao[object_vao].render(mode=mgl.TRIANGLES)
        elif objects[object_vao].type == "line":
            objects_vao[object_vao].render(mode=mgl.LINES)
        elif objects[object_vao].type == "image":
            objects_vao[object_vao].render(mode=mgl.TRIANGLE_STRIP)
        else:
            print(f"Error draw object {object_vao}: unknown type({objects[object_vao].type})")
    objects_last = objects

    fps_counter += 1

    current_time = glfw.get_time()
    fps = 1.0 / (current_time - last_time)
    last_time = current_time

    if Keys.if_pressed(engine.KEY_ESCAPE):
        sys.exit()
    # Для вывода в заголовок окна каждые 0.2 сек
    if current_time % 0.2 < 0.016:  # ~5 раз в секунду
        print(f"FPS: {fps:.2f}")


    current_time_t = time.time()
    delta_time_t = current_time - last_time
    last_time_t = current_time

    # Ограничение FPS
    if delta_time_t < frame_time_t:
        if fps > TARGET_FPS-5:
            time.sleep(frame_time_t - delta_time_t -one_cadr_time/4.1)
            pass
        elif fps < TARGET_FPS-15:
            time.sleep(frame_time_t - delta_time_t - one_cadr_time / 3.0)
            pass
        else:
            time.sleep(frame_time_t - delta_time_t - one_cadr_time / 1)
            pass

    # В update() пишешь код твоей игры в цикле
    update()

glfw.terminate()