import kfp
import kfp.dsl as dsl
from kfp.dsl import container_component, ContainerSpec, pipeline


TOP_K = 100
TEST_SIZE = 0.3
RANDOM_STATE = 42
BASE_IMAGE = "python:3.10-slim"


@container_component
def preprocessing():
    return ContainerSpec(
        image=BASE_IMAGE,
        command=["sh", "-c"],
        args=[
            "pip install --no-cache-dir pandas numpy scikit-learn "
            "python-dotenv requests beautifulsoup4 && "
            "python -m TopKselection.preprocessing"
        ],
    )


@container_component
def scoring():
    return ContainerSpec(
        image=BASE_IMAGE,
        command=["sh", "-c"],
        args=[
            "pip install --no-cache-dir pandas numpy scikit-learn && "
            "python -m TopKselection.scoring"
        ],
    )


@container_component
def supervised(top_k: int):
    return ContainerSpec(
        image=BASE_IMAGE,
        command=["sh", "-c"],
        args=[
            "pip install --no-cache-dir pandas numpy scikit-learn "
            "xgboost joblib && "
            "PCT=$(python -c \"import sys; k=int(sys.argv[1]); print(k//5)\" $0) && "
            "python -m TopKselection.supervised --top-percentile=$PCT",
            top_k,
        ],
    )


@container_component
def clustering():
    return ContainerSpec(
        image=BASE_IMAGE,
        command=["sh", "-c"],
        args=[
            "pip install --no-cache-dir pandas numpy scikit-learn "
            "matplotlib seaborn && "
            "python -m TopKselection.clustering"
        ],
    )


@container_component
def association_rules():
    return ContainerSpec(
        image=BASE_IMAGE,
        command=["sh", "-c"],
        args=[
            "pip install --no-cache-dir pandas numpy mlxtend && "
            "python -m TopKselection.association_rules"
        ],
    )


@container_component
def dashboard():
    return ContainerSpec(
        image=BASE_IMAGE,
        command=["sh", "-c"],
        args=[
            "pip install --no-cache-dir pandas numpy scikit-learn "
            "matplotlib seaborn streamlit plotly httpx python-dotenv && "
            "streamlit run dashboard/app.py "
            "--server.port=8501 --server.address=0.0.0.0"
        ],
    )


@pipeline(
    name="smart-ecommerce-ml-pipeline",
    description="Pipeline ML pour le scoring, clustering et analyse de produits e-commerce",
)
def smart_ecommerce_pipeline(
    top_k: int = TOP_K,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    prep = preprocessing()
    prep.set_caching_options(False)

    score = scoring()
    score.after(prep)
    score.set_caching_options(False)

    sup = supervised(top_k=top_k)
    sup.after(score)
    sup.set_caching_options(False)

    clust = clustering()
    clust.after(score)
    clust.set_caching_options(False)

    assoc = association_rules()
    assoc.after(clust)
    assoc.set_caching_options(False)

    dash = dashboard()
    dash.after(assoc)
    dash.set_caching_options(False)


if __name__ == "__main__":
    output_path = __file__.replace(".py", ".yaml")
    kfp.compiler.Compiler().compile(
        pipeline_func=smart_ecommerce_pipeline,
        package_path=output_path,
    )

    print(f"[kfp] Pipeline compile: {output_path}")
    print("[kfp] Etapes: preprocessing -> scoring -> supervised/clustering -> association -> dashboard")
