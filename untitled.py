#!/usr/bin/env python
# visit https://tool.lu/pyc/ for more information
import rsoft.rsalgorithm as rsalgo
import rsoft.rspytools as rspy
import numarray
import random
from math import floor

class Genetic(rsalgo.RS_Algorithm):             #类

    '''Genetic algorithm.
       遗传算法
  This class implements a genetic algorithm which optimizes a multi-dimensional
  parameter space.  It starts with a generation randomly populated with a
  user-specified number of children, ranks them based on a metric, and then
  applies several breeding operators in order to form the next generation.
  这个类实现了一个优化多维参数空间的遗传算法。它从随机填充了用户指定数量的孩子的一代开始，
  根据一个度量标准对他们进行排序，然后应用几个繁殖操作以形成下一代。

  The following breeding operators are available:
  以下是可供选择的育种经营办法:
  class Genetic (rsalgo.RS_Algorithm):   #类
    - Elitism:
    精英主义
        This type of operator copies a user-specifed percenatage of the best
        ranked to the next generation.  The symbol ga_pc_reproduce controls
        this percentage.
        这种类型的操作符将用户指定的最佳排名百分比复制到下一代。
        符号ga_pc_reproduction控制这个百分比。

    - Mutation:
    基因突变
        Two types of mutation are available:        两种类型的基因突变可用
         -Basic mutation- This type of mutation simple changes the value of a
            variable to another value within the min/max range specified for
            that variable by the user.
        -基本突变-这种类型的突变简单地将一个变量的值更改为用户为该变量指定的最小/最大范围内的另一个值。

         -Percentage mutation- This type of mutation changes the value of a
            variable by multiplying it by a factor randomly determined to lie
            within the range [-mutation_factor, mutation_factor] where the
            value of the parameter mutation_factor is specified by the user.
        -百分比突变-这种类型的突变改变一个变量的值乘以一个随机确定的因素在范围
        [-mutation_factor, mutation_factor]，其中参数mutation_factor的值是由用户指定的。

            If the modified value lies outside the user-specified range for
            that variable, the value is automatically adjusted to lie within
            the range.
        如果修改后的值位于用户为该变量指定的范围之外，则该值将自动调整到该范围内。

        The variable ga_mutation_type sets the type of mutation to be used.
        A value of 0 corresponds to Basic mutation.  A non-zero value
        corresponds to Percentage mutation, and the value sets the
        mutation_factor described above.
        变量ga_mutation_type设置要使用的突变类型。A的值对应Basic突变。
        非零值对应于Percentage mutation，该值设置上面描述的mutation_factor。

    - Crossover:     交叉
        Two types of crossover are avaiable:    两种类型的交叉是可用的
         -Single-point crossover- This type of crossover takes two parents and
            crosses thier genetic information at a single random point.  For
            example, the parents P1= 00000000 and P2= 11111111 could be crossed
            at the third position to obtain a child 00011111.  This type of
            crossover allows for the evolution of both individual variables,
            and sequences of variables.  A good example that benifits from this
            type of crossover would be a taper optimization.
            -单点交叉-这种类型的交叉采取两个父母和交叉他们的遗传信息在一个单一的随机点。例如，
            P1= 00000000和P2= 11111111可以在第三位相交，得到一个孩子00011111。
            这种类型的交叉允许个体变量和变量序列的进化。从这种交叉中受益的一个很好的例子是锥形优化。

         -Random crossover- This type of crossover does not attempt to preserve
            sequences of variables, and randomly crosses two parents to produce
            a child.  For example, the parents P1= 00000000 and P2= 11111111
            could be crossed to produce a child 01001101.  This is basically
            equivilent to crossing the parents at a random number of points.
            This is useful when the sequence of variables is not important and
            should not be attempted to be preserved.
            -随机交叉-这种类型的交叉不试图保留变量序列，并随机交叉两个父母产生一个孩子。例如，
            P1= 00000000和P2= 11111111可以杂交产生一个孩子01001101。
            这基本上相当于在随机数量的点上跨越父结点。E当变量序列不重要且不应试图保留时，这是很有用的。

        The variable ga_cross_type setse the type of crossover to use.  A value
        of 0 corrseponds to Single-point crossover, and a value of 1
        corresponds to Random crossover.
        变量ga_cross_type设置要使用的交叉类型。0表示单点交叉，1表示随机交叉。

  The algorithm supports the following symbol table variables to control its
  options:
  算法支持以下符号表变量来控制其选项:
    algo_ga_num_children:    (int)    number of children per generation (def=0)
    每代的子代数(def=0)
    algo_ga_pc_reproduce:    (int)    percentage of children copied (def=10)
    被复制的孩子的百分比(def=10)
    algo_ga_pc_mutate:       (int)    percentage of children to mutate (def=2)
    儿童发生突变的百分比(def=2)
    algo_ga_pc_kill:         (int)    the percentage of children to remove (def=5)
    需要移除的孩子的百分比(def=5)
    algo_ga_cross_type:      (int)    crossover type (see above) (def=0)
    交叉类型(见上)(def=0)
    algo_ga_mutation_factor: (double) mutation type (See above) (def=0)
    突变类型(见上文)(def=0)
    algo_ga_seed:            (int)    seed for random number generation (def=0)
    随机数生成的种子(def=0)
  References:
  引用
    write me!
    
  '''
    
    def __init__(self, *args):
        rsalgo.RS_Algorithm.__init__(self, *args)
        self.random_box = random.Random()
        self.report_notes('\nThe Maxsteps control determines the number of generations.\n'
                          #Maxsteps控制决定生成的量
                          '\nThe algorithm supports the following symbol table variables to control its options:\n  '
                          #该算法支持以下符号表变量来控制它的选项:
                          'algo_ga_num_children [40]:\n  '
                          'algo_ga_pc_reproduce [10]:\n  algo_ga_pc_kill [2]:\n  algo_ga_pc_mutate [5]:\n  algo_ga_seed [0]:\n  '
                          'algo_ga_cross_type [0]:\n  algo_ga_mutation_factor [0.0]:\n  ')

    
    def declare_allowed_numbers_as_string(self):
        return ('M=2-100', 'N=0')
    #声明允许数量作为字符串
    
    def declare_allowed_number_of_variables(self):
        return (2, 100)
    #声明允许数量作为变量
    
    def declare_allowed_number_of_initial_values(self, num_vars):
        return (0, 0)
    #声明允许数量初值

    def declare_default_maxsteps_and_convergence(self):
        return (20, 1e-05)
    #声明默认maxsteps_and_convergence
    
    def declare_documentation(self):
        return (rsalgo.RSALGO_DOCS_RSMANUAL, '5fglobalmultivariableoptimizationbygeneticalgorithmsgenetic.htm')
    #声明当地的多变量的优化
    
    def declare_is_parallelizable(self):
        return (1,)
    #声明

    
    def go(self):
        mygo(self)

    
    def gen_coef(self, minval, maxval):
        return self.random_box.uniform(minval, maxval)



def mygo(algo):
    num_coef = algo.query_indep_vars_number()
    num_children = algo.query_symbol_table_int('algo_ga_num_children', 40)
    pc_reproduce = algo.query_symbol_table_int('algo_ga_pc_reproduce', 10)
    pc_kill = algo.query_symbol_table_int('algo_ga_pc_kill', 2)
    pc_mutate = algo.query_symbol_table_int('algo_ga_pc_mutate', 5)
    rseed = algo.query_symbol_table_int('algo_ga_seed', 0)
    cross_type = algo.query_symbol_table_int('algo_ga_cross_type', 0)
    mutation_factor = algo.query_symbol_table_double('algo_ga_mutation_factor', 0)
    verbosity = algo.query_verbosity()
    maxsteps = algo.query_maxsteps()
    mintol = algo.query_convergence_tolerance()
    if not rseed or rseed:
        pass
    algo.random_box.seed(None)
    children = numarray.zeros([
        num_children,
        num_coef], 'd')
    old_children = numarray.zeros([
        num_children,
        num_coef], 'd')
    child = numarray.zeros(num_coef, 'd')
    f = numarray.zeros(num_children, 'd')
    rank = numarray.zeros(num_children)
    output = numarray.zeros(3)
    num_reproduce = num_children * pc_reproduce / 100
    num_mutate = num_children * pc_mutate / 100
    num_kill = num_children * pc_kill / 100
    if not num_reproduce:
        num_reproduce = 1
    if not num_mutate:
        num_mutate = 1
    if not num_kill:
        num_kill = 1
    algo.report_plot_data_initialize(3, 0, mintol, [
        'Minimum output',
        'Average output',
        'Maximum output'])
    algo.report_algorithm_comment('Found %d variables...\nUsing %d children...\n' % (num_coef, num_children))
    algo.report_algorithm_comment('Each generation: %d reproduce, %d mutate, %d die\n' % (num_reproduce, num_mutate, num_kill))
    algo.report_algorithm_comment('Using crossover type: %d\n' % cross_type)
    algo.report_algorithm_comment('Using mutation factor: %f\n' % mutation_factor)
    for k in range(0, num_coef):
        minval = algo.query_indep_var_minvalue(k)
        maxval = algo.query_indep_var_maxvalue(k)
        if minval > maxval:
            report_fatal_error('Error:  Min > Max for variable %d.', k)
        for j in range(0, num_children):
            children[(j, k)] = algo.gen_coef(minval, maxval)
        
    
    for steps in range(maxsteps):
        if algo.query_stop():
            return None
        if None > 1:
            algo.report_algorithm_comment('\nCreating generation %i...\n' % steps)
            if verbosity > rsalgo.RSALGO_VERBOSITY_MEDIUM:
                comment = ''
                for j in range(0, num_kill):
                    comment += 'Killed child %d.\n' % rank[-1 - j]
                
                algo.report_algorithm_comment(comment)
                comment = ''
                for j in range(0, num_reproduce):
                    comment += 'Kept child %d.\n' % rank[j]
                
                algo.report_algorithm_comment(comment)
            num_mutate_done = 0
            for i in range(0, num_mutate):
                rc1 = int(round(floor(algo.random_box.uniform(num_reproduce, num_children - num_kill))))
                rpos = int(round(floor(algo.random_box.uniform(0, num_coef))))
                minval = algo.query_indep_var_minvalue(rpos)
                maxval = algo.query_indep_var_maxvalue(rpos)
                if rspy.almost_zero(mutation_factor):
                    children[(rank[rc1], rpos)] = algo.gen_coef(minval, maxval)
                    if verbosity > rsalgo.RSALGO_VERBOSITY_MEDIUM:
                        algo.report_algorithm_comment('Mutated child %d, coef %d.\n' % (rank[rc1], rpos))
                    
                rmf = float(algo.random_box.uniform(-mutation_factor, mutation_factor))
                children[(rank[rc1], rpos)] = rmf * children[(rank[rc1], rpos)]
                if children[(rank[rc1], rpos)] < minval:
                    children[(rank[rc1], rpos)] = minval
                if children[(rank[rc1], rpos)] > maxval:
                    children[(rank[rc1], rpos)] = maxval
                if verbosity > rsalgo.RSALGO_VERBOSITY_MEDIUM:
                    algo.report_algorithm_comment('Mutated child %d, coef %d with factor %f.\n' % (rank[rc1], rpos, rmf))
                    continue
            old_children = children.copy()
            for j in range(num_reproduce, num_children):
                curchild = rank[j]
                rc1 = int(round(floor(algo.random_box.uniform(num_reproduce, num_children - num_kill))))
                rc2 = int(round(floor(algo.random_box.uniform(num_reproduce, num_children - num_kill))))
                while rc2 == rc1:
                    rc2 = int(round(floor(algo.random_box.uniform(num_reproduce, num_children - num_kill))))
                parents = [
                    rc1,
                    rc2]
                if cross_type == 0:
                    rpos = int(round(floor(algo.random_box.uniform(0, num_coef))))
                    children[(curchild, 0:rpos)] = old_children[(parents[0], 0:rpos)]
                    children[(curchild, rpos:)] = old_children[(parents[1], rpos:)]
                    if verbosity > rsalgo.RSALGO_VERBOSITY_MEDIUM:
                        algo.report_algorithm_comment('Crossed child %d from %d and %d at coef %d.\n' % (curchild, rc1, rc2, rpos))
                    
                for pos in range(0, num_coef):
                    rp = int(round(floor(algo.random_box.uniform(0, 2))))
                    children[(curchild, pos)] = old_children[(parents[rp], pos)]
                
                if verbosity > rsalgo.RSALGO_VERBOSITY_MEDIUM:
                    algo.report_algorithm_comment('Bred child %d from %d and %d with random crossover.\n' % (curchild, rc1, rc2))
                    continue
        algo.report_algorithm_comment('\nSimulating generation %i:\n' % steps)
        f = algo.find_f_avs_as(list(children.flat), num_children, num_coef)
        if verbosity > rsalgo.RSALGO_VERBOSITY_MEDIUM:
            print 'metric results:', f
        old_best_value = f[rank[0]]
        rank = numarray.argsort(f)
        output = [
            f[rank[0]],
            numarray.sum(f) * 1 / len(f),
            f[rank[-1]]]
        algo.report_plot_data(steps, output)
        algo.report_algorithm_comment('Average output for generation %i was %f\n' % (steps, output[1]))
        algo.report_algorithm_progress(steps, 1, '\nChild %i has min value of %f so far...\n' % (rank[0], output[0]))
        if output[0] < old_best_value:
            child[0:num_coef] = children[(rank[0], 0:num_coef)]
            algo.preserve_configuration(list(child.flat), output[0], num_coef)
            algo.report_algorithm_comment('New optimal design saved (child %i).\n' % rank[0])
            continue