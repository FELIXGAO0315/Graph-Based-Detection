from meridian.resolution import address_key, match_registry, name_key, person_key, phone_key, resolve_duplicates


def test_transliterated_names_match_but_surnames_stay_apart():
    assert name_key("曹 Medical Supplies Co., Ltd.") == name_key("Cao Medical Supplies Co., Ltd.")
    assert name_key("周 Medical Supplies Co., Ltd.") != name_key("Cao Medical Supplies Co., Ltd.")


def test_duplicate_markers_and_legal_forms_are_stripped():
    assert name_key("Wauters Advisors NV (branch)") == name_key("Wauters Advisors NV")
    assert name_key("Hayes Clinical Research Center Corp. (use this one)") == name_key("Hayes Clinical Research Center Corp.")
    assert name_key("Roldan Healthcare Trading S.A. de C.V. (2)") == name_key("Roldan Healthcare Trading S.A. de C.V.")


def test_address_key_ignores_order_case_and_abbreviations_but_keeps_units():
    a = address_key("Suite 307, Camino Zaira Aranda 53, Alava, 50312", "GT")
    b = address_key("CAMINO ZAIRA ARANDA 53 STE 307 ALAVA 50312", "GT")
    c = address_key("Camino Zaira Aranda 53, Suite 308, Alava, 50312", "GT")
    assert a == b
    assert a != c


def test_person_and_phone_keys():
    assert person_key("Dr Brandon Brown") == person_key("Brown Brandon")
    assert phone_key("+34 921068387") == phone_key("921 068 387")
    assert phone_key("12") is None


def test_registry_matching_reproduces_id_and_name_links(bundle):
    m = match_registry(bundle)
    acc = m[m["status"] == "accepted"]
    assert acc[acc["method"] == "registry_no"]["record_id"].nunique() == 145
    assert acc[acc["method"] == "lei"]["record_id"].nunique() == 65
    assert acc[acc["method"] == "name_jurisdiction"]["record_id"].nunique() >= 114
    # Ambiguous name matches are kept as candidates, never accepted.
    assert not m[m["status"] == "candidate"]["record_id"].isin(acc["record_id"]).any()


def test_duplicates_merge_on_identifiers_not_on_names_alone(bundle):
    emap, pairs = resolve_duplicates(bundle)
    ent = dict(zip(emap["record_id"], emap["entity_id"]))
    assert ent["TP-6CEAE2EE"] == ent["TP-3454D2CF"]          # same registry number
    assert ent["TP-547DF2BA"] != ent["TP-B720E1FC"]          # same name, different registry numbers
    cand = pairs[pairs["relation"] == "IDENTITY_CANDIDATE"]
    assert len(cand) > 0 and (cand["confidence"] < 0.5).all()
