/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package ai.djl.serving.ensemble;

import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.ndarray.NDList;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import java.util.concurrent.CompletableFuture;

/** A {@code Translator} that provides preprocessing and postprocessing for ensemble models. */
public class EnsembleTranslator implements Translator<Input, Output> {

    private Node.Sequence seq;

    EnsembleTranslator(Node.Sequence seq) {
        this.seq = seq;
    }

    /** {@inheritDoc} */
    @Override
    public Batchifier getBatchifier() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Output processOutput(TranslatorContext ctx, NDList list) {
        return (Output) ctx.getAttachment("output");
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, Input input) throws Exception {
        CompletableFuture<Output> future = seq.predict(ctx, input);
        Output output = future.get();
        ctx.setAttachment("output", output);
        return null;
    }
}
