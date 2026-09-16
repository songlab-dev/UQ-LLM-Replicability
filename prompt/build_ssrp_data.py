"""Build the 21-study SSRP analysis table from the D2 and D3 source files.

D3 supplies original and pooled-replication statistics; D2 supplies target
sample sizes. ``CITATIONS`` maps the source study numbers to the references
in Camerer et al. (2018). ``Replicate (R)`` is the pooled same-direction
significance indicator, while the other replication criteria are retained as
separate columns.

Run from repo root: python prompt/build_ssrp_data.py
Writes: data/ssrp_data_cleaned.csv
"""
import os

import pandas as pd

D2_CSV = "data/D2 - ReplicationSampleSizes.csv"
D3_DTA = "data/D3 - ReplicationResults.dta"
OUT_CSV = "data/ssrp_data_cleaned.csv"

# Study Num -> (title, authors, journal, volume, pages, year), transcribed from
# Camerer et al. (2018) references 16-36 (https://doi.org/10.1038/s41562-018-0399-z).
CITATIONS = {
    1: ("Incidental haptic sensations influence social judgments and decisions",
        "Ackerman, J. M., Nocera, C. C. & Bargh, J. A.", "Science", "328", "1712-1715", 2010),
    2: ("Body cues, not facial expressions, discriminate between intense positive and negative emotions",
        "Aviezer, H., Trope, Y. & Todorov, A.", "Science", "338", "1225-1229", 2012),
    3: ("Affirmative action policies promote women and do not harm efficiency in the laboratory",
        "Balafoutas, L. & Sutter, M.", "Science", "335", "579-582", 2012),
    4: ("Experimental evidence for the influence of group size on cultural complexity",
        "Derex, M., Beugin, M.-P., Godelle, B. & Raymond, M.", "Nature", "503", "389-391", 2013),
    5: ("Memory's penumbra: episodic memory decisions induce lingering mnemonic biases",
        "Duncan, K., Sadanand, A. & Davachi, L.", "Science", "337", "485-487", 2012),
    6: ("Analytic thinking promotes religious disbelief",
        "Gervais, W. M. & Norenzayan, A.", "Science", "336", "493-496", 2012),
    7: ("Avoiding overhead aversion in charity",
        "Gneezy, U., Keenan, E. A. & Gneezy, A.", "Science", "346", "632-635", 2014),
    8: ("Cooperating with the future",
        "Hauser, O. P., Rand, D. G., Peysakhovich, A. & Nowak, M. A.", "Nature", "511", "220-223", 2014),
    9: ("Lab experiments for the study of social-ecological systems",
        "Janssen, M. A., Holahan, R., Lee, A. & Ostrom, E.", "Science", "328", "613-617", 2010),
    10: ("Retrieval practice produces more learning than elaborative studying with concept mapping",
         "Karpicke, J. D. & Blunt, J. R.", "Science", "331", "772-775", 2011),
    11: ("Reading literary fiction improves theory of mind",
         "Kidd, D. C. & Castano, E.", "Science", "342", "377-380", 2013),
    12: ("The social sense: susceptibility to others' beliefs in human infants and adults",
         "Kovács, Á. M., Téglás, E. & Endress, A. D.", "Science", "330", "1830-1834", 2010),
    13: ("Washing away postdecisional dissonance",
         "Lee, S. W. S. & Schwarz, N.", "Science", "328", "709", 2010),
    14: ("Thought for food: imagined consumption reduces actual consumption",
         "Morewedge, C. K., Huh, Y. E. & Vosgerau, J.", "Science", "330", "1530-1533", 2010),
    15: ("Inequality and visibility of wealth in experimental social networks",
         "Nishi, A., Shirado, H., Rand, D. G. & Christakis, N. A.", "Nature", "526", "426-429", 2015),
    16: ("Why testing improves memory: mediator effectiveness hypothesis",
         "Pyc, M. A. & Rawson, K. A.", "Science", "330", "335", 2010),
    17: ("Writing about testing worries boosts exam performance in the classroom",
         "Ramirez, G. & Beilock, S. L.", "Science", "331", "211-213", 2011),
    18: ("Spontaneous giving and calculated greed",
         "Rand, D. G., Greene, J. D. & Nowak, M. A.", "Nature", "489", "427-430", 2012),
    19: ("Some consequences of having too little",
         "Shah, A. K., Mullainathan, S. & Shafir, E.", "Science", "338", "682-685", 2012),
    20: ("Google effects on memory: cognitive consequences of having information at our fingertips",
         "Sparrow, B., Liu, J. & Wegner, D. M.", "Science", "333", "776-778", 2011),
    21: ("Just think: the challenges of the disengaged mind",
         "Wilson, T. D. et al.", "Science", "345", "75-77", 2014),
}

RENAME = {
    "study": "Study Num",
    "sref": "Study Reference (O)",
    "type_os": "Test statistic type (O)",
    "stat_os": "Test statistic (O)",
    "n_os": "N (O)",
    "in_os": "Number of individuals (O)",
    "p_os": "P-value (O)",
    "r_os": "Effect size r (O)",
    "r95l_os": "Effect size r, 95% CI lower (O)",
    "r95u_os": "Effect size r, 95% CI upper (O)",
    "r33_os": "Small-telescopes effect size, 33% power (O)",
    "type_rp": "Test statistic type (R)",
    "stat_rp": "Test statistic (R)",
    "n_rp": "N (R)",
    "in_rp": "Number of individuals (R)",
    "p_rp": "P-value (R)",
    "r_rp": "Effect size r (R)",
    "r95l_rp": "Effect size r, 95% CI lower (R)",
    "r95u_rp": "Effect size r, 95% CI upper (R)",
    "pow_rp": "Power (R)",
    "rep_mr_rp": "Meta-analysis significant",
    "rep_st_rp": "Effect size > small telescope threshold (R)",
    "rep_pi_rp": "Effect size within prediction interval (R)",
    "rep_bf_rp": "Replication Bayes factor > 1 (R)",
    "n75": "Target N for 75% power (R)",
    "n50": "Target N for 50% power (R)",
}

COLUMN_ORDER = [
    "Study Num", "Study Title (O)", "Authors (O)", "Journal (O)", "Volume (O)", "Pages (O)",
    "Year (O)", "Study Reference (O)",
    "Test statistic type (O)", "Test statistic (O)", "N (O)", "Number of individuals (O)",
    "P-value (O)", "Effect size r (O)", "Effect size r, 95% CI lower (O)",
    "Effect size r, 95% CI upper (O)", "Small-telescopes effect size, 33% power (O)",
    "Test statistic type (R)", "Test statistic (R)", "N (R)", "Number of individuals (R)",
    "P-value (R)", "Effect size r (R)", "Effect size r, 95% CI lower (R)",
    "Effect size r, 95% CI upper (R)", "Power (R)",
    "Replicate (R)", "Meta-analysis significant",
    "Effect size > small telescope threshold (R)", "Effect size within prediction interval (R)",
    "Replication Bayes factor > 1 (R)",
    "Target N for 75% power (R)", "Target N for 50% power (R)",
]


def main():
    reader = pd.io.stata.StataReader(D3_DTA, convert_categoricals=False)
    d3 = reader.read()
    d3["study"] = d3["study"].astype(int)

    d2 = pd.read_csv(D2_CSV)[["study", "n75", "n50"]]

    out = d3.merge(d2, on="study", how="left")

    citation_cols = ["Study Title (O)", "Authors (O)", "Journal (O)", "Volume (O)", "Pages (O)", "Year (O)"]
    out[citation_cols] = out["study"].map(CITATIONS).apply(pd.Series)
    out["Replicate (R)"] = out["rep_sr_rp"].map({1: "yes", 0: "no"})

    out = out.rename(columns=RENAME)[COLUMN_ORDER]
    out = out.sort_values("Study Num").reset_index(drop=True)

    int_cols = ["N (O)", "Number of individuals (O)", "N (R)", "Number of individuals (R)",
                "Meta-analysis significant", "Effect size > small telescope threshold (R)",
                "Effect size within prediction interval (R)", "Replication Bayes factor > 1 (R)"]
    out[int_cols] = out[int_cols].astype(int)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    print(f"{OUT_CSV}: {len(out)} studies "
          f"({(out['Replicate (R)'] == 'yes').sum()} yes / {(out['Replicate (R)'] == 'no').sum()} no)")


if __name__ == "__main__":
    main()
