import distutils # 用于将字符串转换为布尔值
import json # 用于处理 JSON 数据
import os

from types import SimpleNamespace as Namespace # 将 SimpleNamespace 类别名为 Namespace


def getattr_recursive(obj, s):
    """ 定义递归函数以获取嵌套对象/属性的属性值 """
    if isinstance(s, list):
        split = s
    else:
        split = s.split('/')
    # 如果分割后的长度大于1，则递归调用本函数
    # 否则返回获取到的属性值
    return getattr_recursive(getattr(obj, split[0]), split[1:]) if len(split) > 1 else getattr(obj, split[0])


def setattr_recursive(obj, s, val):
    """ 定义递归函数以设置嵌套对象/属性的属性值 """
    if isinstance(s, list):
        split = s
    else:
        split = s.split('/')
    # 如果分割后的长度大于1，则递归调用本函数
    # 否则设置属性值
    return setattr_recursive(getattr(obj, split[0]), split[1:], val) if len(split) > 1 else setattr(obj, split[0], val)


def generate_config(params, file_path):
    """ 定义生成配置文件的函数 """
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    print("正在保存配置文件")
    f = open(file_path, "w")
    # 将配置参数转换为 JSON 格式的字符串
    json_data = json.dumps(params.__dict__, default=lambda o: o.__dict__, indent=4)
    f.write(json_data)
    f.close()


def read_config(config_path):
    """ 定义读取配置文件的函数 """
    print('正在解析配置文件', config_path, '并将其传递到主函数')
    # 从配置文件中读取 JSON 格式的字符串
    json_data = open(config_path, "r").read()
    # 将 JSON 字符串转换为对象，并使用 Namespace 类来创建对象
    return json.loads(json_data, object_hook=lambda d: Namespace(**d))

def override_params(params, overrides):
    """ 定义覆盖配置参数的函数 """
    assert (len(overrides) % 2 == 0)
    for k in range(0, len(overrides), 2):
    # 获取需要覆盖的属性的旧值
        oldval = getattr_recursive(params, overrides[k])
    # 如果旧值是布尔类型，则将字符串转换为布尔值
        if type(oldval) == bool:
            to_val = bool(distutils.util.strtobool(overrides[k + 1]))
        else:
            # 将字符串转换为与旧值相同类型的值
            to_val = type(oldval)(overrides[k + 1])
            # 设置属性的新值
            setattr_recursive(params, overrides[k], to_val)
            # 输出覆盖参数的信息
            print("正在覆盖参数", overrides[k], "从", oldval, "到", to_val)
    return params

def get_bool_user(message, default: bool):
    """ 定义获取布尔用户输入的函数 """
    if default:
        default_string = '[Y/n]'
    else:
        default_string = '[y/N]'
    # 从用户获取输入并将字符串转换为布尔值
    resp = input('{} {}\n'.format(message, default_string))
    try:
        if distutils.util.strtobool(resp):
            return True
        else:
            return False
    except ValueError:
        # 如果无法将输入的字符串转换为布尔值，则返回默认值
        return default