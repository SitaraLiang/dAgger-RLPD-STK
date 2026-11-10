# dAgger-RLPD-STK

Mini-project 2: DAgger vs symmetric sampling in Super Tux Kart

## Description

The goal of this project is to investigate the relative performance of DAgger and symmetric sampling (from RLPD) in Super Tux Kart driving agents.

## Running the experiments

Run each algorithm independently with a required random seed:

```sh
uv run stk-dagger --seed 0
uv run stk-rlpd --seed 0
```

Use `uv run stk-dagger --help` or `uv run stk-rlpd --help` to see the
algorithm-specific options.

### Remote RLPD runs

Set the machines, seeds, and default remote checkout path in
`scripts/remote_config.sh`, then launch one seed per machine:

```sh
./scripts/launch_remote.sh
```

Additional options are forwarded to `stk-rlpd` on every machine:

```sh
./scripts/launch_remote.sh --rlpd-timesteps 500000 --track abyss
```

The launcher changes to `/Vrac/21500050/dAgger-RLPD-STK` on every remote
machine before starting the run. Runs use tmux sessions named
`rlpd-seed-<seed>` and write results beneath `results/stk/rlpd`. Stop all
configured remote sessions with `./scripts/close_remote.sh`.

Each run also writes its console output to
`results/remote_logs/rlpd-seed-<seed>.log` on its remote machine.

To run DAgger, you will need an expert policy. You can use any policy available in this project:

https://github.com/osigaud/pystk_project/tree/master/src/agents

To run symmetric sampling, you will need a strong dataset. You can generate it using the agents available on the link above.

Your study should focus on a single agent alone on a track and trying to perform one lap as fast as possible, without taking care of opponent-related items.

Using the rl-mind toolkit is recommended, but not mandatory. You may use another RL library, such as Stable baselines 3, Tianshu or CleanRL. 

Your report should contain your empirical study, a more conceptual comparison is also welcome. Think of clearly specifying your methodology and all the hyper-parameters you used to favor reproducibility. Think also of trying to establish statistically valid conclusions. You may also provide a link to a public github project with your sources.


## Evaluation:

The project is on 5 points, from 0 for no report, 3 for standard work and 5 for outstanding work. It must be uploaded on this page on friday october 9, 23:59. Late projects are penalized by 1 point per day of delay, starting from the first minute. Reports longer than 4 pages will be penalized too.


The report should be a pdf named name1_name2.pdf where name1 and name2 are the names of the authors. It should not be longer than 5 pages and written in english. Figures should comply with The figure checklist. Using LLMs to improve formulation is welcome, using it to generate thoughts about the work is strongly discouraged. Do not forget you are here to learn how to think by yourselves.


