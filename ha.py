import logging
import time
import warnings

import numpy as np
import torch
from pymoo.core.algorithm import Algorithm
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.evaluator import Evaluator
from pymoo.core.individual import Individual
from pymoo.core.population import Population
import os

from pymoo.core.variable import Real
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.rnd import FloatRandomSampling

os.environ["OMP_NUM_THREADS"] = "1"
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from scipy.optimize import minimize as scipy_minimize

class NoOpEvaluator(Evaluator):
    def _eval(self, problem, pop, return_values_of, **kwargs):
        # 不做任何评估，假设你已经在别处完成评估
        pass
class HA(Algorithm):
    def __init__(self, method="Adam", pop_size=100,niche_num=3, mutation_rate=1,inherit_rate = 1.0,activate_method = True,X = None, **kwargs):
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
        self.X = X
        self.activate_method = activate_method
        self.inherit_rate = inherit_rate
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
        self._rng = np.random.default_rng(self.seed)
        #不让 PyMoo 自动评估个体
        self.evaluator = NoOpEvaluator()
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
        self.FEs = 0


    def _initialize_infill(self):
        """初始化种群"""
        print("初始化种群...")  # 调试信息
        # 生成初始种群
        if self.X is None:
            print("self.X is None")
            pop_x = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        else:pop_x = self.X
        # 一次性批量评估
        pop_f, pop_cv = self.evaluate_fitness_cv_batch(pop_x)

        # 保存 cv 信息
        self.pop_cv = np.array(pop_cv).reshape(-1, 1)
        pop_f = np.array(pop_f).reshape(-1, 1)
        print("pop.shape =", pop_x.shape)
        print("fit.shape =", pop_f.shape)
        print("cv.shape =", self.pop_cv.shape)

        return Population.new("X", pop_x, "F", pop_f)


    def _infill(self):
        """进化到下一代"""
        print(f"从Generation {self.n_gen-1}进化到Generation {self.n_gen}..")  # 调试信息
        # 获取当前种群
        pop = self.pop.get("X")
        fit = self.pop.get("F")
        cv =  self.pop_cv

        # 计算当前代的最佳可行解

        feasible_indices = np.where(cv <= 0)[0]

        if len(feasible_indices) > 0:
            # 从可行解中找到 f 最小的下标
            best_index = feasible_indices[np.argmin(fit[feasible_indices])]
        else:
            # 没有可行解，从所有个体中找到 cv 最小的下标
            best_index = np.argmin(cv)

        print("best的index是:", best_index)
        print("{:<6} | {:<8} | {:>13.4f} |  {:>13.10f} ".format
              (self.n_gen - 1, self.problem.fes, float(fit[best_index]), float(cv[best_index])))
        print("best在pop_cv中的cv",cv[best_index])

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


        return self.pop

    def extract_function_from_problem(self,x):
        """
        从Problem对象中提取目标函数值
        """
        self.FEs += 1
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

    def evaluate_fitness_cv(self,x):#计算fitness和cv

        if np.any(x > self.ub) or np.any(x < self.lb):
            print(x)
            raise ValueError("x out of bounds")
        self.FEs += 1
        x = np.atleast_2d(x)
        out = {}
        self.problem._evaluate(x, out)
        fitness = out["F"]
        if hasattr(self.problem, "evaluate") and self.problem.has_constraints():
            result = out["G"]
            G = np.atleast_1d(result)
            cv = float(np.sum(np.maximum(0, G)))
        else:
            cv = 0.0
        return  fitness , cv

    def evaluate_fitness_cv_batch(self, X):  # X: shape = (N, dim)
        out = {}
        self.problem._evaluate(X, out)
        fitness = out["F"]  # shape=(N,) 或 (N,1)
        G = out.get("G", None)
        # print(G.shape)
        if G is None:
            # 没有约束，直接返回零数组
            cv = np.zeros((len(X), 1))  # 假设X是样本数组，保证cv是二维列向量
        else:
            G = np.array(G)
            if G.ndim == 1:
                # 一维数组，转成二维列向量 (N,) -> (N,1)
                G = G.reshape(-1, 1)
            elif G.ndim == 0:
                # 标量，转成 (1,1)
                G = G.reshape(1, 1)
            # 计算违反约束的总和
            cv = np.sum(np.maximum(0, G), axis=1).reshape(-1, 1)

        return fitness.reshape(-1, 1), cv
    def _step_ha(self, pop, fit,cv):
        """HA算法的一步进化"""
        """用于种群内排序"""
        def constraint_sort_key(fitness, cv):
            # 返回一个元组：(是否违反约束，违反度，适应度)
            # 违反约束 -> cv>0 -> True，越大越差
            # 优先级：先cv=0，再fit小
            return (cv > 0, cv, fitness)

        previous_pop = pop
        previous_fit = fit
        previous_cv = cv
        # 聚类和学习
        pop, fit, cv, elite_id = self._clustering_and_learning(pop, fit,cv)
        '''
        '''
        '''
        '''
        if not self.check_bounds(pop):
            raise ValueError
        '''
        '''
        '''
        '''
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

        # 通过重写的后代
        num_to_inherit = int(self.inherit_rate * offspring_size)
        offspring = self._inheritance(num_to_inherit, pop, fit)
        # 没有重写的后代
        selected_indices = np.random.choice(self.pop_size, offspring_size - num_to_inherit, replace=False)
        selected_offspring = pop[selected_indices]
        offspring = np.vstack((offspring, selected_offspring))
        ''''
        '''
        if not self.check_bounds(offspring):
            raise ValueError
        print("offspring大小",offspring.shape[0])
        ''''
        '''
        # 变异
        mutate_num = round(offspring_size * self.mutation_rate)
        if mutate_num > 0:
            mutate_id = np.random.choice(offspring_size, mutate_num, replace=False)
            offspring[mutate_id, :] = self._mutate(offspring[mutate_id, :])
        '''
        '''
        if not self.check_bounds(offspring):

            print(f"_mutate越界")
            raise ValueError
        '''
        '''
        # 处理重复个体
        offspring, repeat = np.unique(offspring, axis=0, return_counts=True)


        offspring_fit,offspring_cv = self.evaluate_fitness_cv_batch(offspring)
        offspring_fit.reshape(-1,1)
        offspring_cv.reshape(-1,1)
        # print("offspring_cv",len(offspring_cv),"offspring_fit",len(offspring_fit))

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
        '''
        '''
        new_pop = np.vstack((pop, offspring))
        new_fit = np.vstack((fit, offspring_fit))
        '''
        '''
        # print(cv.shape)  # (N, 9)
        # print(offspring_cv.shape)  # (M, 1)
        '''
        '''
        new_cv = np.vstack((cv, offspring_cv))

        # print("new_pop",len(new_pop),"new_fit",len(new_fit),"new_cv",len(new_cv))

        # 对个体进行排序
        sorted_indices = sorted(range(len(new_pop)),
                                key=lambda i: constraint_sort_key(new_fit[i, 0], new_cv[i, 0]))

        # 选出前pop_size个

        selected_indices = sorted_indices[:self.pop_size]
        # selected_indices = []
        # for _ in range(self.pop_size):
        #     # 随机抽 tour_size 个索引
        #     competitors = np.random.choice(sorted_indices, size=2, replace=False)
        #     # 比较胜者：取排序靠前的那个
        #     winner = min(competitors, key=lambda i: sorted_indices.index(i))
        #     selected_indices.append(winner)

        new_pop = new_pop[selected_indices]
        new_fit = new_fit[selected_indices]
        new_cv = new_cv[selected_indices]

        sorted_indices = sorted(range(len(new_pop)),
                                key=lambda i: constraint_sort_key(new_fit[i, 0], new_cv[i, 0]))
        new_pop = new_pop[sorted_indices]
        new_fit = new_fit[sorted_indices]
        new_cv = new_cv[sorted_indices]

        for i in range(5):
            print(f"fit={new_fit[i, 0]:.4f}, cv={new_cv[i, 0]:.4f}")

        offspring_no_repeat, _= np.unique(new_pop, axis=0, return_counts=True)
        print("种群中不重复的个体数量", offspring_no_repeat.shape[0])
        if offspring_no_repeat.shape[0] <= 2:
            self.termination.force_termination = True
        return new_pop, new_fit, new_cv

    def _clustering_and_learning(self, pop, fit,cv):
        if not self.check_bounds(pop):
            print("传入的pop越界")
            raise ValueError

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

            print("best_individual_idx =", best_individual_idx)
            before = self.problem.fes
            print("local_search之前：",pop[best_individual_idx, :],fit[best_individual_idx],cv[best_individual_idx])

            if self.activate_method :
                new_solution = self._local_search(pop[best_individual_idx, :],fit[best_individual_idx]+10*cv[best_individual_idx])#这里要使用自适应部分的损失函数
            else:
                new_solution = pop[best_individual_idx, :]

            after = self.problem.fes
            print(f"local_search消耗了{after-before}次仿真")
            if np.any(new_solution > self.ub) or np.any(new_solution < self.lb):
                print(new_solution)
                raise ValueError("x out of bounds")
            # print("after:", new_solution[:5],self.problem.evaluate(new_solution),self.calculate_cv(new_solution))
            # 更新个体
            pop[best_individual_idx, :] = new_solution
            if self.activate_method:
                new_fitness,new_cv = self.evaluate_fitness_cv(new_solution)
            else: new_fitness,new_cv = fit[best_individual_idx],cv[best_individual_idx]
            fit[best_individual_idx, 0] = new_fitness
            cv[best_individual_idx, 0] = new_cv
            elite_id.append(best_individual_idx)
            print("local_search之后：", pop[best_individual_idx, :], new_fitness,new_cv)
        return pop, fit ,cv, elite_id

    def _local_search(self, x0,y0):
        """局部搜索"""
        bounded_methods = ["L-BFGS-B", "TNC", "SLSQP", "Powell", "trust-constr"]
        if not self.check_bounds([x0]):
            print("传入的x0越界")
            raise ValueError

        def fun(x):
            result = self.evaluate_fitness_cv(x)
            fitness = result[0]

            cv = result[1]

            alpha = abs(fitness).mean() * 10

            return fitness + alpha * cv

        # def approximate_gradient(f, x,y,lb=None, ub=None,  eps=1e-6):
        #     grad = np.zeros_like(x)
        #     fx = f(x)
        #     for i in range(len(x)):
        #         x_eps = x.copy()
        #         x_eps[i] += eps
        #         # # 保证在边界范围内
        #         if lb is not None:
        #             x_eps[i] = max(x_eps[i], lb[i])
        #         if ub is not None:
        #             x_eps[i] = min(x_eps[i], ub[i])
        #         grad[i] = (f(x_eps) - fx) / eps
        #     return grad
        #
        #
        # def adam_optimize(f, x0,y0 ,lb,ub,lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, max_iter=self.dim, grad_eps=1e-6):
        #     # 梯度估计
        #     x = x0.copy()
        #     y = y0.item()
        #     m = np.zeros_like(x)
        #     v = np.zeros_like(x)
        #     for t in range(1, max_iter + 1):
        #         # 使用有限差分估计梯度
        #         grad = approximate_gradient(f, x, y,lb,ub, eps=grad_eps)
        #         # Adam 更新
        #         m = beta1 * m + (1 - beta1) * grad
        #         v = beta2 * v + (1 - beta2) * (grad ** 2)
        #         m_hat = m / (1 - beta1 ** t)
        #         v_hat = v / (1 - beta2 ** t)
        #         x -= lr * m_hat / (np.sqrt(v_hat) + eps)
        #         x = np.clip(x,lb,ub)
        #     return x
        def approximate_gradient(f, x, lb=None, ub=None, eps=1e-6):
            grad = np.zeros_like(x)
            fx = f(x)
            for i in range(len(x)):
                x_eps = x.copy()
                x_eps[i] += eps
                # 保证在边界内扰动
                if lb is not None:
                    x_eps[i] = min(max(x_eps[i], lb[i]), ub[i])
                grad[i] = (f(x_eps) - fx) / eps
            return grad

        def project_gradient(grad, x, lb, ub):
            grad_proj = grad.copy()
            for i in range(len(x)):
                if x[i] <= lb[i] and grad[i] < 0:
                    grad_proj[i] = 0
                elif x[i] >= ub[i] and grad[i] > 0:
                    grad_proj[i] = 0
            return grad_proj

        def adam_optimize(f, x0, y0,lb, ub, max_iter=self.dim, lr=0.01,
                          beta1=0.9, beta2=0.999, eps=1e-8,
                          grad_eps=1e-6, tol=1e-6, verbose=False):
            x = x0.copy()
            m = np.zeros_like(x)
            v = np.zeros_like(x)

            for t in range(1, max_iter + 1):
                # 1. 估计梯度（有限差分）
                grad = approximate_gradient(f, x, lb, ub, eps=grad_eps)

                # 2. 可选：梯度投影，处理边界约束
                grad = project_gradient(grad, x, lb, ub)

                # 3. 收敛判断
                if np.linalg.norm(grad) < tol:
                    if verbose:
                        print(f"[Adam] 收敛于第 {t} 次迭代，梯度范数为 {np.linalg.norm(grad):.3e}")
                    break

                # 4. Adam 更新
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad ** 2)
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)

                # 5. 参数更新
                x_new = x - lr * m_hat / (np.sqrt(v_hat) + eps)

                # 6. 投影到边界
                x_new = np.clip(x_new, lb, ub)

                # 7. 变量更新幅度判断（也可以作为收敛条件）
                if np.linalg.norm(x_new - x) < tol:
                    if verbose:
                        print(f"[Adam] 收敛于第 {t} 次迭代，Δx 范数为 {np.linalg.norm(x_new - x):.3e}")
                    x = x_new
                    break

                x = x_new
            return x
        if self.method in bounded_methods:
            bounds = [(self.lb[i], self.ub[i]) for i in range(self.dim)]
            """检查边界"""
            x0 = np.clip(x0, self.lb, self.ub)


            minimize_kwargs = {
                "fun":fun,
                "x0": x0,
                "method": self.method,
                "bounds": bounds
            }

            if self.method in ["L-BFGS-B", "SLSQP", "Powell", "trust-constr"]:
                minimize_kwargs["options"] = {"maxiter": 1}
            else:
                minimize_kwargs["options"] = {"maxfun":100,"disp": True}


            result = scipy_minimize(**minimize_kwargs)
            print("消耗了:",result.nfev,"迭代了:",result.nit,"message:",result.message)
            result_x = np.clip(result.x, self.lb, self.ub)

            return result_x



        elif self.method == "Adam":
            # print(fun(x0),y0.item())
            x0 = adam_optimize(fun,x0,y0,self.lb,self.ub)
            new_solution = x0
            return new_solution



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
        '''
        '''
        out_of_bounds = np.any(offspring < self.lb, axis=1) | np.any(offspring > self.ub, axis=1)

        if np.any(out_of_bounds):
            print("越界个体索引：", np.where(out_of_bounds)[0])
            print("对应个体：", offspring[out_of_bounds])
            raise ValueError("有 offspring 个体越界")
        '''
        '''
        return offspring
    def check_bounds(self,x):
        out_of_bounds = np.any(x < self.lb, axis=1) | np.any(x > self.ub, axis=1)

        if np.any(out_of_bounds):
            print("越界个体索引：", np.where(out_of_bounds)[0])
            print("对应个体：", x[out_of_bounds])

            return False
        else:
            # print("没有越界")
            return True

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
            success = False
            # 尝试每个方向
            mutated[i, :] = x.flatten()
            for k in order:
                direction = dir_sign[k] * dir_vector[:, index_vec[k]].reshape(-1, 1)
                noise = np.random.randn(*direction.shape) * 0.01
                direction += noise
                candidate = x + self.step_size * direction


                candidate = np.clip(candidate, self.lb.reshape(-1,1), self.ub.reshape(-1,1))

                if self._is_feasible(candidate, tol):
                    success = True
                    mutated[i, :] = candidate.flatten()
                    break
            if not success:
                print(f"第 {i} 个个体没有找到可行变异方向，保留原解")
        out_of_bounds = np.any(mutated < self.lb, axis=1) | np.any(mutated > self.ub, axis=1)
        '''
        '''
        if np.any(out_of_bounds):
            print("变异越界：")
            print("越界个体索引：", np.where(out_of_bounds)[0])
            print("对应个体：", mutated[out_of_bounds])
            raise ValueError("有 offspring 个体越界")
        '''
        '''
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
    # def _mutate(self, offspring):
    #     """改进的自适应变异"""
    #     # 更温和的步长更新
    #     if self.n_gen <= 2:
    #         self.step_size = 0.1
    #     else:
    #         if self.improvement:
    #             self.step_size = min(0.5, self.step_size * 1.5)  # 更温和的增长
    #         else:
    #             self.step_size = max(1e-4, self.step_size * 0.8)  # 更温和的衰减
    #
    #     mutated = np.zeros_like(offspring)
    #     tol = 1e-6
    #
    #     for i in range(offspring.shape[0]):
    #         x = offspring[i, :].reshape(-1, 1)
    #
    #         # 为每个个体生成不同的变异
    #         mutated[i, :] = self._mutate_individual(x, tol)
    #
    #     return mutated

    def _mutate_individual(self, x, tol):
        """对单个个体进行变异"""
        dim = x.shape[0]

        # 多种变异策略组合
        strategies = [
            self._gaussian_mutation,
            self._directional_mutation,
            self._boundary_mutation
        ]

        # 随机选择变异策略
        strategy = np.random.choice(strategies)
        candidate = strategy(x, tol)

        # 如果变异失败，回退到原解加小噪声
        if not self._is_feasible(candidate, tol):
            noise = np.random.normal(0, self.step_size * 0.1, (dim, 1))
            candidate = np.clip(x + noise,
                                self.lb.reshape(-1, 1),
                                self.ub.reshape(-1, 1))

        return candidate.flatten()

    def _gaussian_mutation(self, x, tol):
        """高斯变异"""
        dim = x.shape[0]
        noise = np.random.normal(0, self.step_size, (dim, 1))
        return np.clip(x + noise,
                       self.lb.reshape(-1, 1),
                       self.ub.reshape(-1, 1))

    def _directional_mutation(self, x, tol):
        """方向性变异"""
        basis, tangent_cone = self._get_directions(self.step_size, x, tol)

        if tangent_cone.shape[1] > 0:
            tangent_cone = tangent_cone[:, np.sum(tangent_cone == 1, axis=0) == 1]

        dir_vector = np.hstack((basis, tangent_cone))

        # 随机选择1-3个方向组合
        n_dirs = np.random.randint(1, min(4, dir_vector.shape[1] + 1))
        selected_dirs = np.random.choice(dir_vector.shape[1], n_dirs, replace=False)

        # 随机权重组合多个方向
        weights = np.random.normal(0, 1, n_dirs)
        combined_direction = np.zeros((x.shape[0], 1))

        for i, dir_idx in enumerate(selected_dirs):
            sign = np.random.choice([-1, 1])
            combined_direction += weights[i] * sign * dir_vector[:, dir_idx].reshape(-1, 1)

        # 归一化
        if np.linalg.norm(combined_direction) > 0:
            combined_direction = combined_direction / np.linalg.norm(combined_direction)

        return np.clip(x + self.step_size * combined_direction,
                       self.lb.reshape(-1, 1),
                       self.ub.reshape(-1, 1))

    def _boundary_mutation(self, x, tol):
        """边界感知变异"""
        dim = x.shape[0]
        lb = self.lb.reshape(-1, 1)
        ub = self.ub.reshape(-1, 1)

        # 计算到边界的距离
        dist_to_lb = x - lb
        dist_to_ub = ub - x

        # 自适应变异强度
        adaptive_step = np.minimum(dist_to_lb, dist_to_ub) * 0.1
        adaptive_step = np.maximum(adaptive_step, self.step_size * 0.01)

        noise = np.random.normal(0, 1, (dim, 1)) * adaptive_step
        return np.clip(x + noise, lb, ub)
