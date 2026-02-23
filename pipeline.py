# Copyright (c) 2026 Zhe Liu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dotenv import load_dotenv
from metaflow import environment  # pyright: ignore[reportAttributeAccessIssue]
from metaflow.decorators import step
from metaflow.flowspec import FlowSpec

load_dotenv()


class NanoRecSysPipeline(FlowSpec):
    """
    End-to-end Machine Learning Pipeline for nanoRecSys
    """

    @environment(vars={"TQDM_DISABLE": "1"})
    @step
    def start(self):
        """Initialize the pipeline."""
        print("Starting the nanoRecSys E2E ML Pipeline...")
        self.retriever_epochs = 300
        self.next(self.process_data)

    @step
    def process_data(self):
        """Run data processing and splitting."""
        print("Step 1: Processing Data...")

        from nanoRecSys.data.build_dataset import (
            prebuild_sequential_files,
            process_data,
        )
        from nanoRecSys.data.splits import create_user_time_split

        process_data()
        create_user_time_split(val_k=1, test_k=1)
        prebuild_sequential_files()

        self.next(self.train_retriever)

    @step
    def train_retriever(self):
        """Train the two-tower retriever model."""
        print("Step 2: Training Retriever...")
        import argparse

        from nanoRecSys.train import main as train_main

        args = argparse.Namespace(
            mode="retriever",
            user_tower_type="transformer",
            epochs=self.retriever_epochs,
            batch_size=128,
            lr=1e-3,
            num_workers=4,
            enable_progress_bar=False,
        )
        train_main(args)

        self.next(self.build_index)

    @step
    def build_index(self):
        """Build FAISS index and embeddings."""
        print("Step 3: Building Embeddings & FAISS Index...")
        from nanoRecSys.indexing.build_embeddings import (
            build_item_embeddings,
            build_user_embeddings,
        )
        from nanoRecSys.indexing.build_faiss_flat import build_flat_index

        build_item_embeddings()
        build_user_embeddings(user_tower_type="transformer")
        build_flat_index()

        self.next(self.mine_negatives)

    @step
    def mine_negatives(self):
        """Mine hard negatives using the trained retriever."""
        print("Step 4: Mining Negatives...")
        from nanoRecSys.training.mine_negatives_sasrec import run_pipeline

        run_pipeline(
            batch_size=128,
            top_k=100,
            skip_top=10,
            sampling_ratio=0.2,
        )

        self.next(self.train_ranker)

    @step
    def train_ranker(self):
        """Train the ranking model."""
        print("Step 5: Training Ranker...")
        import argparse

        from nanoRecSys.train import main as train_main

        args = argparse.Namespace(
            mode="ranker",
            user_tower_type="transformer",
            epochs=5,
            batch_size=2048,
            random_neg_ratio=0.01,
            lr=1e-3,
            item_lr=0.0,
            num_workers=2,
            warmup_steps=500,
            check_val_every_n_epoch=1,
            enable_progress_bar=False,
        )
        train_main(args)

        self.next(self.export_onnx)

    @step
    def export_onnx(self):
        """Export trained models to ONNX."""
        print("Step 6: Exporting models to ONNX...")
        from scripts.export_onnx import main as export_onnx_main

        export_onnx_main()

        self.next(self.end)

    @step
    def end(self):
        """Finish the pipeline."""
        print(
            "Pipeline completed successfully! Models and Index are ready for serving."
        )


if __name__ == "__main__":
    NanoRecSysPipeline()
    # python pipeline.py run
