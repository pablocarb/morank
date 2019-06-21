# morank: multiobjective optimization-based ranking for metabolic pathway design.

Pablo Carbonell, 2019

**Bioretrosynthesis** tools allow determining the metabolic scope connecting a chemical target to the chassis 
organism through heterologous enzymatic transformations encoded into reaction rules. **Pathway enumeration** algorithms 
have been developed in order to determine all viable solutions in the metabolic scope.

For each pathway, several criteria can be used in order to assess its performance like number of enzymatic steps, enzyme efficiency, 
toxicity, theoretical titers and yields, etc. 
All of these are **conflicting objectives** and therefore it is not possible to identify a unique solution unless the user explicitly 
provides some expert criteria for preferencial decision. Defining such preference in the objectives requires a high level of 
context information and expertise that is often hard to achieve.

The routines here address this need in metabolic pathway design through a **multiobjective optimization approach**. 
Rather than asking for a prior preferential decision, the approach consists on analyzing the optimal properties of the 
feasible space of solutions obtained by pathway enumeration. In that way, the progressive Pareto optimal fronts of the solutions
are used in order to rank the pathway subpopulations. Within each group, either user-supplied or a global criteria based on the 
distance to the ideal solution is applied in order to provide the ordered set of pathway solutions.

See the [notebook](Notes.ipynb) for detailed information and examples.
