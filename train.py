import argparse
from src.utils import *
from src.DHMulti.Environment import DHMultiEnvironmentParams, DHMultiEnvironment
from src.CPP.Environment import CPPEnvironmentParams, CPPEnvironment

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="DHMulti", choices=["DHMulti", "CPP"], help="训练任务，选择DHMulit或者CPP")
    parser.add_argument("--resume", default=None, help="从模型恢复训练")
    parser.add_argument("--step", default=0, type=int, help="用于恢复训练指定步数")
    parser.add_argument("--episode", default=0, type=int, help="用于恢复训练指定回合数")
    parser.add_argument("--config", default=None, help="训练参数路径")
    parser.add_argument("--gen_config", action="store_true", help="生成默认的参数文件")
    parser.add_argument("--id", default=None, help="设置训练的ID号")
    args = parser.parse_args()
    return args

def main_cpp(params):
    env = CPPEnvironment(params)
    env.run()
    
def main_dhmulti(params):
    env = DHMultiEnvironment(params)
    env.run()

def create_dirs():
    if not os.path.exists("models"):
        os.makedirs("models")
    if not os.path.exists("logs"):
        os.makedirs("logs")
    if not os.path.exists("eval"):
        os.makedirs("eval")

if __name__ == "__main__":
    create_dirs() # 创建文件夹
    args = get_args()
    if args.gen_config:
        if args.target == "DHMulti":
            generate_config(DHMultiEnvironmentParams(), "config/dhmulti.json")
        elif args.target == "CPP":
            generate_config(CPPEnvironmentParams(), "config/cpp.json")
        else:
            raise ValueError("任务名错误！")
    
    if args.config is None:
        raise ValueError("需要指定参数文件！")
    
    params = read_config(args.config)
    
    if args.resume is not None:
        params.step_count = args.step
        params.episode_count = args.episode
        params.agent_params.resume = args.resume
    
    if args.id is not None:
        params.model_stats_params.save_model = "models/" + args.id
        params.model_stats_params.log_file_name = args.id
        
    if args.target == "DHMulti":
        main_dhmulti(params)
    elif args.target == "CPP":
        main_cpp(params)