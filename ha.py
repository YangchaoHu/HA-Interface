import logging
import time
import warnings

import numpy as np
import torch
from pymoo.core.algorithm import Algorithm
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.individual import Individual
from pymoo.core.population import Population
import os



os.environ["OMP_NUM_THREADS"] = "1"
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from scipy.optimize import minimize as scipy_minimize


class HA(Algorithm):
    def __init__(self, method="Powell", pop_size=100,niche_num=3, mutation_rate=0.3, **kwargs):
        """
        参数:
            method: 局部搜索方法，支持 "L-BFGS-B", "TNC", "SLSQP", "Powell", "trust-constr", "Adam"
            niche_num: 聚类数量
            mutation_rate: 变异率
        """
        super().__init__(**kwargs)
        self.method = method
        self.niche_num = niche_num
        self.mutation_rate = mutation_rate
        self.pop_size = pop_size

        # 算法参数
        self.step_size = 1
        self.improvement = True
        self.fun_cnt = 0
        self.best = float("inf")
        self.best_individual = None

        # 将在_setup中设置的参数
        self.elite_num = None
        self.dim = None
        self.lb = None
        self.ub = None

    def _setup(self, problem, **kwargs):
        """设置算法参数"""
        # print("设置算法参数...")  # 调试信息
        # print("problem的lb:",problem.xl)
        # print("problem的ub",problem.xu)
        super()._setup(problem, **kwargs)

        # 从problem中获取参数
        self.dim = problem.n_var

        # 如果xl和xu是数组
        if hasattr(problem.xl, '__len__'):
            self.lb = np.array(problem.xl)
            self.ub = np.array(problem.xu)
        # 如果xl和xu是标量，则创建与问题维度相同的数组
        else:
            self.lb = np.full(problem.n_var, problem.xl)
            self.ub = np.full(problem.n_var, problem.xu)

        # 精英数量
        self.elite_num = self.dim


    def _initialize_infill(self):
        """初始化种群"""
        # print("初始化种群...")  # 调试信息

        # 生成初始种群
        pop_x = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        pop_f = np.array([self.extract_function_from_problem(x) for x in pop_x]).reshape(-1, 1)
        self.pop_cv = np.array([self.calculate_cv(x) for x in pop_x]).reshape(-1, 1)
        # print("_initialize_infill:cv.shape:",self.pop_cv.shape)


        return Population.new("X", pop_x, "F", pop_f)


    def _infill(self):
        """进化到下一代"""
        # print(f"Generation {self.n_gen}: Processing next generation...")  # 调试信息
        # 获取当前种群
        pop = self.pop.get("X")
        fit = self.pop.get("F")
        cv =  self.pop_cv
        # print("_infill:cv.shape:",cv.shape)
        # 更新最优解
        # best_idx = np.argmin(fit[:, 0])
        # current_best = fit[best_idx, 0]
        best_idx = 0    #pop里面已经有序
        current_best = fit[best_idx, 0]

        if current_best < self.best:
            self.best = current_best
            self.best_individual = pop[best_idx].copy()
            self.improvement = True
            # print(f"New best found: {self.best}")
        else:
            self.improvement = False

        # 执行HA算法的一代步
        new_pop, new_fit, new_cv= self._step_ha(pop, fit, cv)
        # print("unique:", np.unique(new_pop, axis=0).shape)
        # 创建新的Population对象并更新当前种群
        self.pop = Population.new("X", new_pop, "F", new_fit)
        self.pop_cv = new_cv

        cv_min = np.min(cv)
        cv_avg = np.mean(cv)
        f_avg = np.mean(fit)
        f_min = np.min(fit)

        print("{:<6} | {:<8} | {:>13.6E} | {:>13.6E} | {:>13.10f} | {:>13.10f}".format(
            self.n_gen, self.evaluator.n_eval, cv_min, cv_avg, f_avg, f_min
        ))
        return self.pop

    def extract_function_from_problem(self,x):
        """
        从Problem对象中提取目标函数值
        """
        x = np.atleast_2d(x)
        out = {}
        self.problem._evaluate(x, out)
        return out["F"]

    def calculate_cv(self,x):
        if hasattr(self.problem, "evaluate")  and self.problem.has_constraints():
            result = self.problem.evaluate(x, return_values_of=["G"])
            G = np.atleast_1d(result)
            cv = float(np.sum(np.maximum(0, G)))
        else:
            cv = 0.0
        return cv

    def _step_ha(self, pop, fit,cv):
        """HA算法的一步进化"""
        """用于种群内排序"""
        def constraint_sort_key(fitness, cv):
            # 返回一个元组：(是否违反约束，违反度，适应度)
            # 违反约束 -> cv>0 -> True，越大越差
            # 优先级：先cv=0，再fit小
            return (cv > 0, cv, fitness)

        # 聚类和学习
        pop, fit, cv, elite_id = self._clustering_and_learning(pop, fit,cv)

        # 对种群内个体进行排序
        sorted_indices = sorted(range(len(pop)),
                                key=lambda i: constraint_sort_key(fit[i, 0], cv[i, 0]))
        pop = pop[sorted_indices]
        fit = fit[sorted_indices]
        cv = cv[sorted_indices]

        # 添加全局最优个体到精英中
        global_elite_id = np.argsort(fit[:, 0])[:self.elite_num].tolist()
        elite_id.extend(global_elite_id)

        # 去重并保持原始顺序
        elite_id = [elite_id[i] for i in sorted(np.unique(elite_id, return_index=True)[1])]

        # 计算后代数量
        offspring_size = self.pop_size - len(elite_id)

        # 生成后代
        offspring = self._inheritance(offspring_size, pop, fit)

        # 变异
        mutate_num = round(offspring_size * self.mutation_rate)
        if mutate_num > 0:
            mutate_id = np.random.choice(offspring_size, mutate_num, replace=False)
            offspring[mutate_id, :] = self._mutate(offspring[mutate_id, :])

        # 处理重复个体
        offspring, repeat = np.unique(offspring, axis=0, return_counts=True)
        offspring_fit = np.array([self.extract_function_from_problem(x) for x in offspring]).reshape(-1, 1)
        offspring_cv = np.array([self.calculate_cv(x) for x in offspring]).reshape(-1, 1)


        # 恢复重复个体
        repeat -= 1
        repeat_index = np.nonzero(repeat)[0]
        if len(repeat_index) > 0:
            offspring = np.vstack((offspring,
                                   np.repeat(offspring[repeat_index, :], repeat[repeat_index], axis=0)))
            offspring_fit = np.vstack((offspring_fit,
                                       np.repeat(offspring_fit[repeat_index, :], repeat[repeat_index], axis=0)))
            offspring_cv = np.vstack((offspring_cv,
                                      np.repeat(offspring_cv[repeat_index, :], repeat[repeat_index], axis=0)))

        # 合并当前代和后代
        new_pop = np.vstack((pop, offspring))
        new_fit = np.vstack((fit, offspring_fit))
        '''
        '''
        # print(cv.shape)  # (N, 9)
        # print(offspring_cv.shape)  # (M, 1)
        '''
        '''
        new_cv = np.vstack((cv, offspring_cv))  # 精英假设CV=0



        # 对个体进行排序
        sorted_indices = sorted(range(len(new_pop)),
                                key=lambda i: constraint_sort_key(new_fit[i, 0], new_cv[i, 0]))

        # 选出前pop_size个
        selected_indices = sorted_indices[:self.pop_size]

        new_pop = new_pop[selected_indices]
        new_fit = new_fit[selected_indices]
        new_cv = new_cv[selected_indices]
        return new_pop, new_fit, new_cv

    def _clustering_and_learning(self, pop, fit,cv):
        """聚类和局部学习"""
        elite_id = []

        # K-means聚类
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        if self.n_gen == 0:
            kmeans = KMeans(n_clusters=self.niche_num, n_init=1, random_state=42)
        else:
            # 使用前面的精英作为初始中心
            init_centers = pop[:min(self.niche_num, pop.shape[0]), :]
            kmeans = KMeans(n_clusters=self.niche_num, init=init_centers, n_init=1, random_state=42)

        labels = kmeans.fit_predict(pop)

        # 对每个聚类进行局部搜索
        for i in range(self.niche_num):
            cluster_idx = np.where(labels == i)[0]
            if len(cluster_idx) == 0:
                continue

            # 找到聚类中的最优个体
            best_individual_idx = cluster_idx[0]

            # 局部搜索
            # print("before:", pop[best_individual_idx, :5],self.problem.evaluate(pop[best_individual_idx, :]),self.calculate_cv(pop[best_individual_idx, :]))
            new_solution, new_value = self._local_search(pop[best_individual_idx, :])
            # print("after:", new_solution[:5],self.problem.evaluate(new_solution),self.calculate_cv(new_solution))
            # 更新个体
            pop[best_individual_idx, :] = new_solution
            fit[best_individual_idx, 0] = new_value
            cv[best_individual_idx, 0] = self.calculate_cv(new_solution)
            elite_id.append(best_individual_idx)

        return pop, fit ,cv, elite_id

    def _local_search(self, x0):
        """局部搜索"""
        bounded_methods = ["L-BFGS-B", "TNC", "SLSQP", "Powell", "trust-constr"]

        if self.method in bounded_methods:
            bounds = [(self.lb[i], self.ub[i]) for i in range(self.dim)]
            """检查边界"""
            x0 = np.clip(x0, self.lb, self.ub)
            def fun(x):
                return self.extract_function_from_problem(x)+10*self.calculate_cv(x)

            minimize_kwargs = {
                "fun":fun,
                "x0": x0,
                "method": self.method,
                "bounds": bounds
            }

            if self.method in ["L-BFGS-B", "SLSQP", "Powell", "trust-constr"]:
                minimize_kwargs["options"] = {"maxiter": 1}

            # try:
                result = scipy_minimize(**minimize_kwargs)
                result_x = np.clip(result.x, self.lb, self.ub)

                return result_x, self.extract_function_from_problem(result_x)
            # except Exception as e:
            #     # 捕获异常并打印
            #     logging.error("An error occurred during optimization: %s", str(e))
            #
            #     return x0, self.problem.evaluate(x0)

        elif self.method == "Adam":
            class FunctionWrapper:
                def __init__(self, blackbox_func):
                    self.blackbox_func = blackbox_func
                    self.call_count = 0

                def __call__(self, x_torch):
                    self.call_count += 1

                    # 将PyTorch张量转换为numpy数组
                    x_numpy = x_torch.detach().cpu().numpy()

                    # 调用黑箱函数
                    result = self.blackbox_func(x_numpy)
                    # 将结果转换回PyTorch张量
                    return torch.tensor(result, dtype=x_torch.dtype, device=x_torch.device)
            # try:
            x = torch.tensor(x0, dtype=torch.float32, requires_grad=True)
            optimizer = torch.optim.Adam([x], lr=0.01)
            wrapped_func = FunctionWrapper(self.extract_function_from_problem)
            for _ in range(self.dim):
                optimizer.zero_grad()
                loss = wrapped_func(x)
                if isinstance(loss, np.ndarray):
                    loss = torch.tensor(loss[0], requires_grad=True)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    x.clamp_(min=torch.tensor(self.lb), max=torch.tensor(self.ub))

            new_solution = x.detach().numpy()
            new_value = self.problem.evaluate(new_solution)

            return new_solution, new_value

            # except Exception as e:
            #     # 捕获异常并打印
            #     logging.error("An error occurred during optimization: %s", str(e))
            #     return x0, self.problem.evaluate(x0)

        else:
            raise ValueError(f"不支持的优化方法: {self.method}")

    def _inheritance(self, offspring_size, pop, fit):
        """基于适应度的遗传操作"""
        # 计算选择概率
        # rank_id = np.argsort(fit[:, 0])
        scores = np.arange(pop.shape[0], 0, -1)

        offspring = np.zeros((offspring_size, self.dim))

        for i in range(offspring_size):
            # 随机选择父母
            parent_size = self.dim
            parent_idx = np.random.choice(pop.shape[0], parent_size, replace=False)
            parent_scores = scores[parent_idx]
            parent_pop = pop[parent_idx, :]

            # 基于适应度的概率选择
            probs = parent_scores / np.sum(parent_scores)

            for j in range(self.dim):
                offspring[i, j] = np.random.choice(parent_pop[:, j], p=probs)

        return offspring

    def _mutate(self, offspring):
        """自适应变异"""
        # 更新步长
        if self.n_gen <= 2:
            self.step_size = 1
        else:
            if self.improvement:
                self.step_size = min(1, self.step_size * 4)
            else:
                self.step_size = max(1e-6, self.step_size / 4)

        mutated = np.zeros_like(offspring)
        tol = 1e-3

        for i in range(offspring.shape[0]):
            x = offspring[i, :].reshape(-1, 1)

            # 获取搜索方向
            basis, tangent_cone = self._get_directions(self.step_size, x, tol)

            # 合并方向
            if tangent_cone.shape[1] > 0:
                tangent_cone = tangent_cone[:, np.sum(tangent_cone == 1, axis=0) == 1]

            dir_vector = np.hstack((basis, tangent_cone))
            n_basis = basis.shape[1]
            n_tangent = tangent_cone.shape[1]
            n_total = n_basis + n_tangent

            # 构造方向索引和符号
            index_vec = np.hstack((np.arange(n_basis), np.arange(n_basis),
                                   np.arange(n_basis, n_total), np.arange(n_basis, n_total)))
            dir_sign = np.hstack((np.ones(n_basis), -np.ones(n_basis),
                                  np.ones(n_tangent), -np.ones(n_tangent)))

            # 随机排列方向
            order = np.random.choice(len(index_vec), len(index_vec), replace=False)

            # 尝试每个方向
            mutated[i, :] = x.flatten()
            for k in order:
                direction = dir_sign[k] * dir_vector[:, index_vec[k]].reshape(-1, 1)
                candidate = x + self.step_size * direction

                if self._is_feasible(candidate, tol):
                    mutated[i, :] = candidate.flatten()
                    break

        return mutated

    def _get_directions(self, mesh_size, x, tol):
        """获取搜索方向"""
        dim = x.shape[0]
        lb = np.expand_dims(self.lb, axis=1)
        ub = np.expand_dims(self.ub, axis=1)

        # 构造切锥
        I = np.eye(dim)
        active = (np.abs(x - lb) < tol) | (np.abs(x - ub) < tol)
        tangent_cone = I[:, active.flatten()]

        # 构造基础方向
        p = 1 / np.sqrt(mesh_size)
        lower_t = np.tril(np.round((p + 1) * np.random.rand(dim, dim) - 0.5), -1)

        diag_temp = p * np.sign(np.random.rand(dim, 1) - 0.5)
        diag_temp[diag_temp == 0] = p * np.sign(0.5 - np.random.rand())

        diag_t = np.diag(diag_temp.flatten())
        basis = lower_t + diag_t

        # 随机排列
        order = np.random.choice(dim, dim, replace=False)
        basis = basis[order][:, order]

        return basis, tangent_cone

    def _is_feasible(self, x, tol):
        """检查解的可行性"""
        lb = np.expand_dims(self.lb, axis=1)
        ub = np.expand_dims(self.ub, axis=1)

        constraint = max(np.max(x - ub), np.max(lb - x), 0)
        return constraint < tol

