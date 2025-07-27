# POMDP_simulation

Currently, all codes use the small toy example attack graph due to the following setting.
    num = "_small"
To use the other graphs in the paper, please use "_real" and "_two" for the real attack graph and the synthetic attack graph, respectively.

For the attacker type, all codes uses the attacker due to the following setting.
    t = "Inter" 
To simulate a benign user, please use "_Low".

# To run the code with the state-space reduction method
Each argument means the following:
1. state space reduction rate.
2. exploit sapce reduction rate.
3. Directory name to save the output

```
python run_diff_omega.py 0 0 state
python run_diff_omega.py 0.01 0 state
python run_diff_omega.py 0.05 0 state
python run_diff_omega.py 0.1 0 state
python run_diff_omega.py 0.2 0 state
python run_diff_omega.py 0.3 0 state
python run_diff_omega.py 0.4 0 state
python run_diff_omega.py 0.5 0 state
```



# To run the code with the exploit-space reduction method

```
python run_diff_omega.py 0.01 0.01 action
python run_diff_omega.py 0.01 0.05 action
python run_diff_omega.py 0.01 0.1 action
python run_diff_omega.py 0.01 0.2 action
python run_diff_omega.py 0.01 0.3 action
python run_diff_omega.py 0.01 0.4 action
python run_diff_omega.py 0.01 0.5 action
```

# To run the code with different false positive rates

```
python run_diff_fps.py 0.01 0.01 fps_reduced_s_a
```



# To run the code with the different investigation budgets.

```
python run_diff_ib.py 0.01 0.01 budget_reduced_s_a
```

