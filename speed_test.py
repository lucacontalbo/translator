import time
import json
import statistics
import torch

# ----------------------------
# SAMPLE PASSAGE
# ----------------------------
#PASSAGE = (
#    "Prior to restoration work performed between 1990 and 2001, "
#    "the Leaning Tower of Pisa leaned at an angle of 5.5 degrees, "
#    "but the tower now leans at about 3.99 degrees. "
#    "This means the top of the Leaning Tower of Pisa is displaced "
#    "horizontally 3.9 meters from the center."
#)

PASSAGE = (
    "Prima dei lavori di restauro eseguiti tra il 1990 e il 2001, "
    "la Torre pendente di Pisa era inclinata di 5,5 gradi, "
    "ma ora la torre è inclinata di circa 3,99 gradi. "
    "Ciò significa che la sommità della Torre pendente di Pisa è spostata "
    "orizzontalmente di 3,9 metri dal centro."
)

#PASSAGE = """Although the board said on Tuesday that the merger, which had been described by its advisers as “effectively complete” after the April vote, was approved unanimously, two directors who had not attended that meeting later claimed that the version of the agreement circulated to investors was not the one they had signed, and that the liabilities the subsidiary was expected to absorb, if the regulators who were already reviewing the deal did not object, were larger than those disclosed in the memo that the CFO, who replaced Martin after his resignation, said she had only summarized"""
#PASSAGE = """Sebbene il consiglio abbia affermato martedì che la fusione, descritta dai suoi consulenti 
#come "effettivamente completata" dopo il voto di aprile, è stata approvata all'unanimità, 
#due amministratori che non avevano partecipato a quella riunione hanno successivamente affermato che la 
#versione dell'accordo diffusa agli investitori non era quella che avevano firmato 
#e che le passività che la controllata avrebbe dovuto assorbire, se le autorità di 
#regolamentazione che stavano già esaminando l'accordo non avessero sollevato obiezioni, erano maggiori di quelle divulgate nel promemoria 
#che il CFO, che ha sostituito Martin dopo le sue dimissioni, ha affermato di aver solo riassunto."""
#PASSAGE = """Q: Was the investor version final?
#A: No. The version sent to investors was a summary draft, not the executed agreement.

#Q: Did all directors approve it?"""
#PASSAGE = """Approval Review Procedure:
#1. Confirm whether the circulated agreement matches the executed version.
#2. Identify any liabilities omitted from the investor memo.
#3. Write a response to Legal and to CFO."""
#PASSAGE = """Editorial note: Rewrite the following in plain English and avoid legal terminology where possible. The board recorded unanimous approval of the merger, but Laura Reed, who was absent from the meeting, later stated that the version circulated to investors was not the executed agreement."""
#PASSAGE = """Please answer the following question.
# What is the boiling point of nitrogen?"""
#PASSAGE = """In the scriptorium, the codex revealed a dense palimpsestic stratigraphy: under raking light, erased glossae re-emerged beneath the later Carolingian minuscule. The bifolium’s parchment showed cockling near the gutter, while iron-gall corrosion had fissured several rubricated initials. A quire signature, nearly abraded, suggested misbinding during a post-medieval rebinding campaign. Marginal scholia in a cramped protogothic hand clustered around a florilegium excerpt, perhaps for monastic lectio. Pricking and drypoint ruling remained visible along the fore-edge, and the vestigial catchword implied the manuscript once belonged to a larger composite miscellany. Even the endbands hinted at a regional, possibly Insular, provenance.""" # DUMMY inserted in several places, some facts have been discarded
#PASSAGE = """Pursuant to Section 2.03(b) of the Merger Agreement, as amended by Amendment No. 2 and incorporated by reference to Exhibit 10.4, the Company stated that NewCo would assume the specified contingent obligations at Closing. Two former directors, however, said the execution version attached as Ex. 10.4A was not the same draft approved at the April 17 meeting. The Form 8-K describes the discrepancy as immaterial, while internal memorandum FIN/MA-22_rev3 suggests the omitted schedule altered the liability cap."""
#PASSAGE = """Patients carrying the CYP2C192 allele showed lower response rates after eight weeks, although the authors cautioned that the study was not powered for genotype-stratified efficacy claims. A secondary analysis reported higher adherence among participants receiving SMS reminders, which may have confounded the treatment effect. The supplementary appendix lists one subgroup as “CYP2C19∗2/∗17,” while the OCRed repository copy renders the same genotype as “CYP2C192I*17.” The paper describes that discrepancy as clerical, not biological."""
#PASSAGE = """The scraped version reads <p>The ministry said the transfer was authorized in May&nbsp;...</p> and preserves &ldquo;provisional&rdquo;, &sect; 4.2, and <fn id="n3">internal note</fn> inline with the sentence text."""
#PASSAGE = """Bí ó tilẹ̀ jẹ́ pé ìgbìmọ̀ náà sọ ní ọjọ́ Ìṣẹ́gun pé a fọwọ́ sí ìṣọ̀kan ilé-iṣẹ́ náà ní ìfọkànsìn gbogbo lẹ́yìn ìdìbò oṣù Kẹrin, àwọn olùdarí méjì tí wọn kò wà nípàdé náà sọ lẹ́yìn náà pé ẹ̀dà àdéhùn tí a fi ránṣẹ́ sí àwọn olùdókòwò kì í ṣe èyí tí wọ́n ti fọwọ́ sí. Wọ́n tún sọ pé ilé-iṣẹ́ ọmọ yóò ti gba àwọn gbèsè tó pọ̀ ju àwọn tí a ṣàfihàn nínú mémò náà lọ. CFO náà sọ pé òun kan ṣe àkótán àwọn àkọsílẹ̀ àwọn olùdámọ̀ràn, nígbà tí àwọn alábojútó tó ń ṣàyẹ̀wò ìdúnàdúrà náà kò fọwọ́ sí ìtàn kankan.""" # yoruba
#PASSAGE = """Quamquam consilium die Martis dixit coniunctionem post suffragium mensis Aprilis consensu omnium approbatam esse, duo administri qui conventui non interfuerant postea dixerunt versionem pactionis investoribus missam non eam fuisse quam ipsi subscripserant. Illi etiam dixerunt subsidiariam obligationes maiores suscepturam fuisse quam eae quae in memorando patefactae erant. Praefectus pecuniarius dixit se annotationes consultorum tantum summatim rettulisse, dum moderatores transactionem examinantes neutram partem comprobaverunt."""
#PASSAGE = """Sebbene il consiglio abbia affermato martedì che la fusione, descritta dai suoi consulenti come “effettivamente completata” dopo il voto di aprile, è stata approvata all’unanimità, due amministratori che non avevano partecipato a quella riunione hanno successivamente affermato che la versione dell’accordo diffusa agli investitori non era quella che avevano firmato. Hanno inoltre affermato che la controllata avrebbe assunto passività maggiori di quelle indicate nel promemoria. Il direttore finanziario ha detto di essersi limitato a riassumere le note dei consulenti, mentre i regolatori che esaminavano l’operazione non hanno avallato nessuna delle due versioni."""
#PASSAGE = """The public's association of the word with the criminal secret society was perhaps inspired by the 1863 play "I mafiusi di la Vicaria" (it) ("The Mafiosi of the Vicaria") by Giuseppe Rizzotto and Gaspare Mosca. The words mafia and mafiusi are never mentioned in the play. The play is about a Palermo prison gang with traits similar to the Mafia: a boss, an initiation ritual, and talk of umirtà (omertà or code of silence) and "pizzu" (a codeword for extortion money). The play had great success throughout Italy. Soon after, the use of the term "mafia" began appearing in the Italian state's early reports on the group. The word was first documented in 1865 in a report by the prefect of Palermo Filippo Antonio Gualterio (it). The public's association of the word with the criminal secret society was perhaps inspired by the 1863 play "I mafiusi di la Vicaria" (it) ("The Mafiosi of the Vicaria") by Giuseppe Rizzotto and Gaspare Mosca. The words mafia and mafiusi are never mentioned in the play. The play is about a Palermo prison gang with traits similar to the Mafia: a boss, an initiation ritual, and talk of umirtà (omertà or code of silence) and "pizzu" (a codeword for extortion money). The play had great success throughout Italy. Soon after, the use of the term "mafia" began appearing in the Italian state's early reports on the group. The word was first documented in 1865 in a report by the prefect of Palermo Filippo Antonio Gualterio (it). The public's association of the word with the criminal secret society was perhaps inspired by the 1863 play "I mafiusi di la Vicaria" (it) ("The Mafiosi of the Vicaria") by Giuseppe Rizzotto and Gaspare Mosca. The words mafia and mafiusi are never mentioned in the play. The play is about a Palermo prison gang with traits similar to the Mafia: a boss, an initiation ritual, and talk of umirtà (omertà or code of silence) and "pizzu" (a codeword for extortion money). The play had great success throughout Italy. Soon after, the use of the term "mafia" began appearing in the Italian state's early reports on the group. The word was first documented in 1865 in a report by the prefect of Palermo Filippo Antonio Gualterio (it)."""
#PASSAGE = """Although the archivist, who had been warned by the curator, who had been alerted by a donor whose grandfather, after he had inherited a trunk that had been sealed before the war had ended, claimed that the letters inside proved that the portrait, which had long been believed to depict a minor diplomat, had in fact been commissioned by a widow who, because she feared that the court, which had already begun to investigate her family, would confiscate the estate unless its accounts, which her steward said were incomplete, could be made to appear orderly, refused to let anyone, even those whom she trusted, see the final page, continued sorting the papers in silence."""

PASSAGE = """Q: La versione per gli investitori era definitiva?
A: No. La versione inviata agli investitori era una bozza riassuntiva, non l’accordo esecutivo.

Q: Tutti i direttori l’hanno approvata?"""

PASSAGE = """Procedura di revisione dell’approvazione:

Confermare se l’accordo circolato corrisponde alla versione eseguita.
Individuare eventuali passività omesse nel memo per gli investitori.
Redigere una risposta per il reparto legale e per il CFO."""

PASSAGE = """Nello scriptorium, il codice rivelava una densa stratigrafia palinsestica: alla luce radente, le glosse cancellate riemergevano sotto la successiva minuscola carolingia. Il bifoglio mostrava increspature della pergamena vicino alla piega interna, mentre la corrosione da inchiostro ferro-gallico aveva fessurato diverse iniziali rubricate. Una segnatura di fascicolo, quasi abrasa, suggeriva un errato assemblaggio durante una campagna di rilegatura post-medievale. Scolî marginali in una mano protogotica fitta si addensavano attorno a un estratto di florilegio, forse per la lectio monastica. Le forature e la rigatura a punta secca restavano visibili lungo il margine esterno, e la parola di richiamo residua indicava che il manoscritto apparteneva un tempo a una miscellanea composita più ampia. Persino i capitelli di cucitura suggerivano una provenienza regionale, forse insulare."""

PASSAGE = """Ai sensi della Sezione 2.03(b) dell’Accordo di Fusione, come modificato dall’Emendamento n. 2 e incorporato per riferimento all’Allegato 10.4, la Società ha dichiarato che NewCo avrebbe assunto le obbligazioni potenziali specificate al Closing. Due ex amministratori, tuttavia, hanno affermato che la versione esecutiva allegata come Ex. 10.4A non era la stessa bozza approvata nella riunione del 17 aprile. Il Form 8-K descrive la discrepanza come irrilevante, mentre il memorandum interno FIN/MA-22_rev3 suggerisce che l’allegato omesso abbia modificato il tetto di responsabilità."""
#PASSAGE = """Per favore rispondi alla seguente domanda.
#Qual è il punto di ebollizione dell’azoto?"""
#PASSAGE = """I pazienti portatori dell’allele CYP2C192 hanno mostrato tassi di risposta inferiori dopo otto settimane, sebbene gli autori abbiano avvertito che lo studio non era dimensionato per trarre conclusioni sull’efficacia stratificata per genotipo. Un’analisi secondaria ha riportato una maggiore aderenza tra i partecipanti che ricevevano promemoria via SMS, il che potrebbe aver confuso l’effetto del trattamento. L’appendice supplementare elenca un sottogruppo come “CYP2C19∗2/∗17”, mentre la copia del repository ottenuta tramite OCR rende lo stesso genotipo come “CYP2C192I*17”. L’articolo descrive tale discrepanza come un errore materiale, non biologico."""
#PASSAGE = """La versione estratta riporta <p>The ministry said the transfer was authorized in May ...</p> e conserva “provisional”, § 4.2 e <fn id="n3">internal note</fn> in linea con il testo della frase."""

print(PASSAGE)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_RUNS = 1
WARMUP = True


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_gpu_stats():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def gpu_mem():
    if torch.cuda.is_available():
        sync_cuda()
        return {
            "allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 2),
            "reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 2),
            "max_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
            "max_reserved_mb": round(torch.cuda.max_memory_reserved() / 1024**2, 2),
        }
    return None


# ----------------------------
# 1) FLAN PROPOSITIONIZER
# ----------------------------
def load_propositionizer():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    #model_name = "chentong00/propositionizer-wiki-flan-t5-large"
    model_name = "outputs/flan_t5_large_it_prop/"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    ).to(DEVICE)
    model.eval()
    return tokenizer, model


def run_propositionizer_once(tokenizer, model, passage):
    prompt = f"Title: . Section: . Content: {passage}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(DEVICE)

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )
    sync_cuda()
    elapsed = time.perf_counter() - start

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {
        "seconds": elapsed,
        "output_text": text,
        "input_tokens": int(inputs["input_ids"].shape[1]),
        "output_tokens": int(outputs.shape[1]),
    }


def benchmark_propositionizer(passage, num_runs=NUM_RUNS, warmup=WARMUP):
    tokenizer, model = load_propositionizer()
    reset_gpu_stats()

    if warmup:
        _ = run_propositionizer_once(tokenizer, model, passage)
        reset_gpu_stats()

    runs = [run_propositionizer_once(tokenizer, model, passage) for _ in range(num_runs)]
    times = [r["seconds"] for r in runs]

    return {
        "method": "propositionizer_flan_t5_large",
        "runs_seconds": times,
        "median_seconds": statistics.median(times),
        "mean_seconds": statistics.mean(times),
        "input_tokens": runs[0]["input_tokens"],
        "output_tokens": runs[0]["output_tokens"],
        "output_text": runs[0]["output_text"],
        "gpu_mem": gpu_mem(),
    }


# ----------------------------
# 2) STANZA DEPPIPELINE
# ----------------------------
def load_stanza_pipeline():
    import stanza

    return stanza.Pipeline(
        lang="en",
        processors="tokenize,pos,lemma,depparse",
        package="default",
        use_gpu=torch.cuda.is_available(),
        verbose=False,
    )


def extract_props_from_deps(doc):
    """
    Minimal placeholder rules.
    Right now this just returns one proposition per sentence.
    Replace this with your actual dependency-based proposition rules.
    """
    props = []
    for sent in doc.sentences:
        text = sent.text.strip()
        if text:
            props.append(text)
    return props


def run_stanza_once(nlp, passage):
    start_pipeline = time.perf_counter()
    doc = nlp(passage)
    sync_cuda()
    pipeline_seconds = time.perf_counter() - start_pipeline

    start_rules = time.perf_counter()
    propositions = extract_props_from_deps(doc)
    rules_seconds = time.perf_counter() - start_rules

    num_tokens = sum(len(sent.words) for sent in doc.sentences)

    return {
        "pipeline_seconds": pipeline_seconds,
        "rules_seconds": rules_seconds,
        "total_seconds": pipeline_seconds + rules_seconds,
        "num_sentences": len(doc.sentences),
        "num_tokens": num_tokens,
        "propositions": propositions,
    }


def benchmark_stanza(passage, num_runs=NUM_RUNS, warmup=WARMUP):
    nlp = load_stanza_pipeline()
    reset_gpu_stats()

    if warmup:
        _ = run_stanza_once(nlp, passage)
        reset_gpu_stats()

    runs = [run_stanza_once(nlp, passage) for _ in range(num_runs)]

    pipeline_times = [r["pipeline_seconds"] for r in runs]
    rules_times = [r["rules_seconds"] for r in runs]
    total_times = [r["total_seconds"] for r in runs]

    return {
        "method": "stanza_sentsplit_depparse_rules",
        "runs_pipeline_seconds": pipeline_times,
        "runs_rules_seconds": rules_times,
        "runs_total_seconds": total_times,
        "median_pipeline_seconds": statistics.median(pipeline_times),
        "median_rules_seconds": statistics.median(rules_times),
        "median_total_seconds": statistics.median(total_times),
        "mean_total_seconds": statistics.mean(total_times),
        "num_sentences": runs[0]["num_sentences"],
        "num_tokens": runs[0]["num_tokens"],
        "propositions": runs[0]["propositions"],
        "gpu_mem": gpu_mem(),
    }


# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    print("Device:", DEVICE)

    flan_result = benchmark_propositionizer(PASSAGE)
    print("\n=== FLAN propositionizer ===")
    print(json.dumps(flan_result, indent=2, ensure_ascii=False))

    """stanza_result = benchmark_stanza(PASSAGE)
    print("\n=== Stanza pipeline ===")
    print(json.dumps(stanza_result, indent=2, ensure_ascii=False))

    print("\n=== Summary ===")
    print(json.dumps({
        "flan_median_seconds": flan_result["median_seconds"],
        "stanza_median_total_seconds": stanza_result["median_total_seconds"],
        "stanza_median_pipeline_seconds": stanza_result["median_pipeline_seconds"],
        "stanza_median_rules_seconds": stanza_result["median_rules_seconds"],
    }, indent=2))"""
