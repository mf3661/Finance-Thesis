from __future__ import annotations

import pandas as pd

from f001_ps import add_ps
from f002_be import add_be
from f003_acc import add_acc
from f004_absacc import add_absacc
from f005_agr import add_agr
from f006_bm import add_bm
from f007_cfp import add_cfp
from f008_ep import add_ep
from f009_op import add_op
from f010_rsup import add_rsup
from f011_rd_sale import add_rd_sale
from f012_pctacc import add_pctacc
from f013_sgr import add_sgr
from f014_ni import add_ni
from f015_sp import add_sp
from f016_gma import add_gma
from f017_chcsho import add_chcsho
from f018_chato import add_chato
from f019_noa import add_noa
from f020_rna import add_rna
from f021_invest import add_invest
from f022_adm import add_adm
from f023_rdm import add_rdm
from f024_rd import add_rd
from f025_lgr import add_lgr
from f026_cash import add_cash
from f027_cashdebt import add_cashdebt
from f028_pchsale_pchinvt import add_pchsale_pchinvt
from f029_pchsale_pchrect import add_pchsale_pchrect
from f030_pchdepr import add_pchdepr
from f031_pchsale_pchxsga import add_pchsale_pchxsga
from f032_pchgm_pchsale import add_pchgm_pchsale
from f033_roa import add_roa
from f034_roe import add_roe
from f035_roic import add_roic
from f036_egr import add_egr
from f037_pchcapx import add_pchcapx
from f038_grGW import add_grGW
from f039_lev import add_lev
from f040_chtx import add_chtx
from f041_dy import add_dy
from f042_pm import add_pm
from f043_ato import add_ato
from f044_depr import add_depr
from f045_chinv import add_chinv
from f046_chadv import add_chadv
from f047_currat import add_currat
from f048_grcapx import add_grcapx
from f049_pchcurrat import add_pchcurrat
from f050_saleinv import add_saleinv
from f051_pchsaleinv import add_pchsaleinv
from f052_obklg import add_obklg
from f053_chobklg import add_chobklg
from f054_conv import add_conv
from f055_convind import add_convind
from f056_grltnoa import add_grltnoa
from f057_quick import add_quick
from f058_pchquick import add_pchquick
from f059_realestate import add_realestate
from f060_salerec import add_salerec
from f061_salecash import add_salecash
from f062_rdbias import add_rdbias
from f063_capxint import add_capxint
from f064_xadint import add_xadint
from f065_chpm import add_chpm
from f066_operprof import add_operprof
from f067_chdrc import add_chdrc
from f068_hire import add_hire
from f069_herf import add_herf
from f070_cashpr import add_cashpr
from f071_ala import add_ala
from f072_alm import add_alm
from f073_age import add_age
from f074_chempia import add_chempia
from f075_chatoia import add_chatoia
from f076_chpmia import add_chpmia
from f077_pchcapx_ia import add_pchcapx_ia
from f078_divi import add_divi
from f079_divo import add_divo
from f080_mohanram_score import add_mohanram_score
from f081_sin import add_sin
from f082_tang import add_tang
from f083_tb import add_tb
from f084_secured import add_secured
from f085_securedind import add_securedind


def build_annual_features(base_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build annual accounting features in dependency order.
    """

    df: pd.DataFrame = base_df.copy()

    df = add_ps(df)
    df = add_be(df)
    df = add_acc(df)
    df = add_absacc(df)
    df = add_agr(df)
    df = add_bm(df)
    df = add_cfp(df)
    df = add_ep(df)
    df = add_op(df)
    df = add_rsup(df)
    df = add_rd_sale(df)
    df = add_pctacc(df)
    df = add_sgr(df)
    df = add_ni(df)
    df = add_sp(df)
    df = add_gma(df)
    df = add_chcsho(df)
    df = add_chato(df)
    df = add_noa(df)
    df = add_rna(df)
    df = add_invest(df)
    df = add_adm(df)
    df = add_rdm(df)
    df = add_rd(df)
    df = add_lgr(df)
    df = add_cash(df)
    df = add_cashdebt(df)
    df = add_pchsale_pchinvt(df)
    df = add_pchsale_pchrect(df)
    df = add_pchdepr(df)
    df = add_pchsale_pchxsga(df)
    df = add_pchgm_pchsale(df)
    df = add_roa(df)
    df = add_roe(df)
    df = add_roic(df)
    df = add_egr(df)
    df = add_pchcapx(df)
    df = add_grGW(df)
    df = add_lev(df)
    df = add_chtx(df)
    df = add_dy(df)
    df = add_pm(df)
    df = add_ato(df)
    df = add_depr(df)
    df = add_chinv(df)
    df = add_chadv(df)
    df = add_currat(df)
    df = add_grcapx(df)
    df = add_pchcurrat(df)
    df = add_saleinv(df)
    df = add_pchsaleinv(df)
    df = add_obklg(df)
    df = add_chobklg(df)
    df = add_conv(df)
    df = add_convind(df)
    df = add_grltnoa(df)
    df = add_quick(df)
    df = add_pchquick(df)
    df = add_realestate(df)
    df = add_salerec(df)
    df = add_salecash(df)
    df = add_rdbias(df)
    df = add_capxint(df)
    df = add_xadint(df)
    df = add_chpm(df)
    df = add_operprof(df)
    df = add_chdrc(df)
    df = add_hire(df)
    df = add_herf(df)
    df = add_cashpr(df)
    df = add_ala(df)
    df = add_alm(df)
    df = add_age(df)
    df = add_chempia(df)
    df = add_chatoia(df)
    df = add_chpmia(df)
    df = add_pchcapx_ia(df)
    df = add_divi(df)
    df = add_divo(df)
    df = add_mohanram_score(df)
    df = add_sin(df)
    df = add_tang(df)
    df = add_tb(df)
    df = add_secured(df)
    df = add_securedind(df)

    return df