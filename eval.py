import argparse
import numpy as np
from src.DHMulti.Environment import DHMultiEnvironmentParams, DHMultiEnvironment
from src.CPP.Environment import CPPEnvironmentParams, CPPEnvironment
from src.utils import read_config
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def eval_logs(event_path):
    event_acc = EventAccumulator(event_path, size_guidance={'tensors': 100000})
    event_acc.Reload()

    def aver(l):
        length = len(l)
        sum_ = sum([v[2] for v in l])
        return sum_ / length

    for k, v in [[k[5:], aver(event_acc.scalars.Items(k))] for k in event_acc.scalars.Keys()]:
        print(f"{k} : {v}")
    

def mc_dhmulti(args, params: DHMultiEnvironmentParams):
    if args.num_agents is not None:
        num_range = [int(i) for i in args.num_agents]
        params.grid_params.num_agents_range = num_range
        env = DHMultiEnvironment(params)
        env.gen_model_graph() # 构建完整的模型
        env.agent.load_weights(args.weights) # 加载模型权重
        env.eval(int(args.samples), show=args.show)
        eval_logs("logs/" + args.id)

def mc_cpp(args, params: CPPEnvironmentParams):
    env = CPPEnvironment(params)
    env.gen_model_graph() # 构建完整的模型
    env.agent.load_weights(args.weights) # 加载模型权重
    env.eval(int(args.samples), show=args.show)
    eval_logs("logs/" + args.id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True, help='权重路径')
    parser.add_argument('--config', required=True, help='配置参数文件')
    parser.add_argument('--id', required=False, help='输入文件的ID')
    parser.add_argument('--samples', required=True, help='评估进行的回合数量')
    parser.add_argument('--seed', default=None, help="随机种子")
    parser.add_argument('--show', default=False, help="是否展示")
    parser.add_argument('--num_agents', default=None, help='智能体数量，如 12 代表智能体数量范围 [1,2]')
    parser.add_argument("--target", default="DHMulti", choices=["DHMulti", "CPP"], help="训练任务，选择DHMulit或者CPP")

    args = parser.parse_args()

    if args.seed:
        np.random.seed(int(args.seed))

    params = read_config(args.config)

    if args.id is not None:
        params.model_stats_params.save_model = "models/" + args.id
        params.model_stats_params.log_file_name = args.id
    else:
        args.id = params.model_stats_params.log_file_name

    if args.target == "DHMulti":
        mc_dhmulti(args, params)
    elif args.target == "CPP":
        mc_cpp(args, params)
        
