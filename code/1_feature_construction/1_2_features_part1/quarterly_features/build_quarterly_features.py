from __future__ import annotations

import pandas as pd

from f001_chtx import add_chtx
from f002_acc import add_acc
from f003_bm import add_bm
from f004_cfp import add_cfp
from f005_ep import add_ep
from f006_ope import add_ope
from f007_agr import add_agr
from f008_ni import add_ni
from f009_cash import add_cash
from f010_cashdebt import add_cashdebt
from f011_rd import add_rd
from f012_chcsho import add_chcsho
from f013_pctacc import add_pctacc
from f014_sgr import add_sgr
from f015_sp import add_sp
from f016_invest import add_invest
from f017_egr import add_egr
from f018_lev import add_lev
from f019_gma import add_gma
from f020_lgr import add_lgr
from f021_depr import add_depr
from f022_rdm import add_rdm
from f023_rd_sale import add_rd_sale
from f024_chpm import add_chpm
from f025_chato import add_chato
from f026_pm import add_pm
from f027_noa import add_noa
from f028_rna import add_rna
from f029_ato import add_ato
from f030_grltnoa import add_grltnoa
from f031_ala import add_ala
from f032_rsup import add_rsup
from f033_stdacc import add_stdacc
from f034_stdcf import add_stdcf
from f035_roa import add_roa
from f036_roe import add_roe
from f037_roavol import add_roavol
from f038_cinvest import add_cinvest
from f039_nincr import add_nincr
from f040_pscore import add_pscore
from f041_opa import add_opa
from f042_cop import add_cop
from f043_chatoia import add_chatoia
from f044_bm_ia import add_bm_ia
from f045_me_ia import add_me_ia
from f046_cfp_ia import add_cfp_ia
from f047_alm import add_alm
from f048_sgrvol import add_sgrvol
from f049_cashpr import add_cashpr
from f050_mohanram_vol_score import add_mohanram_vol_score


def build_quarterly_features(
    base_quarterly_df: pd.DataFrame,
    annual_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build quarterly accounting features in dependency order.
    """

    df: pd.DataFrame = base_quarterly_df.copy()

    df = add_chtx(df)
    df = add_acc(df)
    df = add_bm(df)
    df = add_cfp(df)
    df = add_ep(df)
    df = add_ope(df)
    df = add_agr(df)
    df = add_ni(df)
    df = add_cash(df)
    df = add_cashdebt(df)
    df = add_rd(df)
    df = add_chcsho(df)
    df = add_pctacc(df)
    df = add_sgr(df)
    df = add_sp(df)
    df = add_invest(df)
    df = add_egr(df)
    df = add_lev(df)
    df = add_gma(df)
    df = add_lgr(df)
    df = add_depr(df)
    df = add_rdm(df)
    df = add_rd_sale(df)
    df = add_chpm(df)
    df = add_chato(df)
    df = add_pm(df)
    df = add_noa(df)
    df = add_rna(df)
    df = add_ato(df)
    df = add_grltnoa(df)
    df = add_ala(df)
    df = add_rsup(df)
    df = add_stdacc(df)
    df = add_stdcf(df)
    df = add_roa(df)
    df = add_roe(df)
    df = add_roavol(df)
    df = add_cinvest(df)
    df = add_nincr(df)
    df = add_pscore(df)
    df = add_opa(df)

    # cop needs annual xpp because there is no quarterly xpp.
    df = add_cop(
        quarterly_df=df,
        annual_df=annual_df,
    )

    df = add_chatoia(df)
    df = add_bm_ia(df)
    df = add_me_ia(df)
    df = add_cfp_ia(df)
    df = add_alm(df)
    df = add_sgrvol(df)
    df = add_cashpr(df)
    df = add_mohanram_vol_score(df)

    return df