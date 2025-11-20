import json
import os
import random
import re
import numpy as np

class JobTemplate:
    def __init__(self, model, command, working_directory, num_steps_arg,
                 needs_data_dir=True, distributed=False):
        self._model = model
        self._command = command
        self._working_directory = working_directory
        self._num_steps_arg = num_steps_arg
        self._needs_data_dir = needs_data_dir
        self._distributed = distributed

    @property
    def model(self):
        return self._model

    @property
    def command(self):
        return self._command

    @property
    def working_directory(self):
        return self._working_directory

    @property
    def num_steps_arg(self):
        return self._num_steps_arg

    @property
    def needs_data_dir(self):
        return self._needs_data_dir

    @property
    def distributed(self):
        return self._distributed

def resnet18(batch_size):
    model = 'ResNet-18 (batch size %d)' % batch_size
    command = 'python3 main.py --data_dir=%s/cifar10'
    command += ' --batch_size %d' % batch_size
    working_directory = 'image_classification/cifar10'
    num_steps_arg = '--num_steps'
    return JobTemplate(model=model, command=command,
                       working_directory=working_directory,
                       num_steps_arg=num_steps_arg, distributed=True)

def resnet50(batch_size):
    model = 'ResNet-50 (batch size %d)' % batch_size
    command = 'python3 main.py -j 8 -a resnet50 -b %d' % batch_size
    command += ' %s/imagenet/'
    working_directory = 'image_classification/imagenet'
    num_steps_arg = '--num_minibatches'
    return JobTemplate(model=model, command=command,
                       working_directory=working_directory,
                       num_steps_arg=num_steps_arg, distributed=True)

def transformer(batch_size):
    model = 'Transformer (batch size %d)' % batch_size
    command = 'python3 train.py -data %s/translation/multi30k.atok.low.pt'
    command += ' -batch_size %d -proj_share_weight' % batch_size
    working_directory = 'translation'
    num_steps_arg = '-step'
    return JobTemplate(model=model, command=command,
                       working_directory=working_directory,
                       num_steps_arg=num_steps_arg, distributed=True)

def lm(batch_size):
    model = 'LM (batch size %d)' % batch_size
    command = 'python main.py --cuda --data %s/wikitext2'
    command += ' --batch_size %d' % batch_size
    working_directory = 'language_modeling'
    num_steps_arg = '--steps'
    return JobTemplate(model=model, command=command,
                       working_directory=working_directory,
                       num_steps_arg=num_steps_arg, distributed=True)

def recommendation(batch_size):
    model = 'Recommendation (batch size %d)' % batch_size
    command = 'python3 train.py --data_dir %s/ml-20m/pro_sg/'
    command += ' --batch_size %d' % batch_size
    working_directory = 'recommendation'
    num_steps_arg = '-n'
    return JobTemplate(model=model, command=command,
                       working_directory=working_directory,
                       num_steps_arg=num_steps_arg)

def a3c():
    model = 'A3C'
    command = ('python3 main.py --env PongDeterministic-v4 --workers 4 '
               '--amsgrad True')
    working_directory = 'rl'
    num_steps_arg = '--max-steps'
    return JobTemplate(model=model, command=command,
                       working_directory=working_directory,
                       num_steps_arg=num_steps_arg,
                       needs_data_dir=False)

def cyclegan():
    model = 'CycleGAN'
    working_directory = 'cyclegan'
    command = ('python3 cyclegan.py --dataset_path %s/monet2photo'
               ' --decay_epoch 0')
    num_steps_arg = '--n_steps'
    return JobTemplate(model=model, command=command,
                       working_directory=working_directory,
                       num_steps_arg=num_steps_arg)

JobTable = []

for batch_size in [16, 32, 64, 128, 256]:
    JobTable.append(resnet18(batch_size))
for batch_size in [16, 32, 64, 128]:
    JobTable.append(resnet50(batch_size))
for batch_size in [16, 32, 64, 128, 256]:
    JobTable.append(transformer(batch_size))
for batch_size in [5, 10, 20, 40, 80]:
    JobTable.append(lm(batch_size))
for batch_size in [512, 1024, 2048, 4096, 8192]:
    JobTable.append(recommendation(batch_size))
# JobTable.append(a3c())
# JobTable.append(cyclegan())

model_names = ["ResNet-18", "ResNet-50", "Transformer", "LM", "Recommendation"]

batch_sizes_range = {
    "ResNet-18": [16, 256],
    "ResNet-50": [16, 128],
    "Transformer": [16, 256],
    "LM": [5, 80],
    "Recommendation": [512, 8192]
}

PHYSICAL_CLUSTER_THROUGHPUT_FILE = "./physical_all.json"

class ProblemDescription:
    def __init__(self, jobs=None, server_num=15):
        self.worker_types = []
        self.physical_throughput_list = self.read_throughput_data(PHYSICAL_CLUSTER_THROUGHPUT_FILE)
        
        self.S = server_num
        self.A = len(self.worker_types)
        
        self.job_types = self.get_job_types()
        
        if jobs is None:
            jobs_num = random.randint(20, 80)
            self.jobs = random.choices(self.job_types, k=jobs_num)
        else:
            self.jobs = jobs if jobs is not None else []
            
        self.J = len(self.jobs)
        
        self.CList = []
        self.Tr = []  # Throughput matrix: J x C x A
        
        if self.jobs:
            self.prepare_problem()
    
    def get_job_types(self):
        job_types = [(job_template.model, 1) for job_template in JobTable]
        return job_types
    
    def add_job(self, job):
        """Add a job to the problem and update the throughput matrix"""
        self.jobs.append(job)
        self.J = len(self.jobs)
        
        # Update combinations and throughput matrix
        self.update_for_new_job(job)
        
        return self.J - 1  # Return the index of the new job
    
    def update_for_new_job(self, new_job):
        """Update combinations and throughput matrix when a new job is added"""
        # Add new single job combination
        self.CList.append((self.J-1,))
        
        # Add new pair combinations with existing jobs
        for i in range(self.J-1):
            self.CList.append((i, self.J-1))
        
        # Update throughput matrix for all jobs
        self.update_throughput_matrix()
    
    def update_throughput_matrix(self):
        """Update the throughput matrix for all jobs"""
        # Reinitialize the throughput matrix
        self.Tr = []
        
        for j in range(self.J):
            Tr_j = np.zeros((len(self.CList), self.A))
            
            for c, combination in enumerate(self.CList):
                if j not in combination:
                    continue
                
                if len(combination) == 1:
                    # Single job case
                    for a in range(self.A):
                        worker = self.worker_types[a]
                        job_key = self.jobs[j]
                        Tr_j[c, a] = self.physical_throughput_list[worker].get(job_key, {}).get('null', 0)
                else:
                    # Job pair case
                    if combination[0] == j:
                        other_job_idx = combination[1]
                    else:
                        other_job_idx = combination[0]
                    
                    for a in range(self.A):
                        worker = self.worker_types[a]
                        job_key = self.jobs[j]
                        other_job_key = self.jobs[other_job_idx]
                        Tr_j[c, a] = self.physical_throughput_list[worker].get(job_key, {}).get(other_job_key, 0)[0]
            
            self.Tr.append(Tr_j)
    
    def read_throughput_data(self, file_name):
        with open(file_name, 'r') as f:
            raw_data = json.load(f)
        
        parsed_data = {}
        for worker_type in raw_data:
            if 'unconsolidated' in worker_type:
                continue
            self.worker_types.append(worker_type)
            
            parsed_data[worker_type] = {}
            for job_type in raw_data[worker_type]:
                key = self.parse_job_type(job_type)
                if key is None:
                    continue
                
                parsed_data[worker_type][key] = {}
                for other_job_type in raw_data[worker_type][job_type]:
                    if other_job_type == 'null':
                        other_key = other_job_type
                    else:
                        other_key = self.parse_job_type(other_job_type)
                        if other_key is None:
                            continue
                    
                    parsed_data[worker_type][key][other_key] = \
                        raw_data[worker_type][job_type][other_job_type]
        
        return parsed_data
    
    def parse_job_type(self, job_type_str):
        match = re.match(r"\(\'(.*)\', (\d+)\)", job_type_str)
        if match is None:
            return None
        model = match.group(1)
        scale_factor = int(match.group(2))
        return (model, scale_factor)
    
    def prepare_problem(self):
        self.generate_combinations()
        self.generate_throughput_matrix()
    
    def generate_combinations(self):
        # Generate all possible job combinations (single and pairs)
        for i in range(self.J):
            for j in range(i, self.J):
                if i == j:
                    self.CList.append((i,))
                else:
                    self.CList.append((i, j))
    
    def generate_throughput_matrix(self):
        for j in range(self.J):
            Tr_j = np.zeros((len(self.CList), self.A))
            
            for c, combination in enumerate(self.CList):
                if j not in combination:
                    continue
                
                if len(combination) == 1:
                    # Single job case
                    for a in range(self.A):
                        worker = self.worker_types[a]
                        job_key = self.jobs[j]
                        Tr_j[c, a] = self.physical_throughput_list[worker].get(job_key, {}).get('null', 0)
                else:
                    # Job pair case
                    if combination[0] == j:
                        other_job_idx = combination[1]
                    else:
                        other_job_idx = combination[0]
                    
                    for a in range(self.A):
                        worker = self.worker_types[a]
                        job_key = self.jobs[j]
                        other_job_key = self.jobs[other_job_idx]
                        Tr_j[c, a] = self.physical_throughput_list[worker].get(job_key, {}).get(other_job_key, 0)[0]
            
            self.Tr.append(Tr_j)
    
    def get_throughput(self, job_idx, combination_idx, accelerator_idx):
        return self.Tr[job_idx][combination_idx, accelerator_idx]
    
    def get_job_count(self):
        return self.J
    
    def get_combination_count(self):
        return len(self.CList)
    
    def get_accelerator_count(self):
        return self.A