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
import ai.djl.ndarray.BytesSupplier;
import ai.djl.ndarray.NDList;
import ai.djl.serving.models.ModelManager;
import ai.djl.serving.wlm.Job;
import ai.djl.serving.wlm.ModelInfo;
import ai.djl.translate.TranslateException;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.Pair;
import ai.djl.util.PairList;
import java.lang.reflect.Constructor;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;

/** An interface that represents an ensemble model node. */
public abstract class Node {

    /**
     * Predicts an item for inference.
     *
     * @param ctx the toolkit for creating the input NDArray
     * @param input the input object
     * @return a {@code CompletableFuture} instance
     * @throws TranslateException if an error occurs during processing input
     */
    public abstract CompletableFuture<Output> predict(TranslatorContext ctx, Input input)
            throws TranslateException;

    protected List<Input> split(Input input) {
        PairList<String, BytesSupplier> pairs = input.getContent();
        List<Input> list = new ArrayList<>(pairs.size());
        for (Pair<String, BytesSupplier> pair : pairs) {
            Input in = new Input();
            in.setProperties(input.getProperties());
            in.add(pair.getKey(), pair.getValue());
            list.add(in);
        }
        return list;
    }

    /** An {@code Node} that contains a sequence of children. */
    final class Sequence extends Node {

        private List<Node> children;

        /** {@inheritDoc} */
        @Override
        public CompletableFuture<Output> predict(TranslatorContext ctx, Input input)
                throws TranslateException {
            CompletableFuture<Output> ret = null;
            try {
                for (Node node : children) {
                    ret = node.predict(ctx, input);
                    input = ret.get();
                }
            } catch (InterruptedException | ExecutionException e) {
                throw new TranslateException(e);
            }

            return ret;
        }
    }

    /**
     * A {@code Node} whose children form a parallel branch in the network and are combined to
     * produce a single output.
     */
    final class Parallel extends Node {

        private List<Node> children;
        private boolean splitInputs;
        private boolean mergeOutputs;

        /** {@inheritDoc} */
        @Override
        public CompletableFuture<Output> predict(TranslatorContext ctx, Input input)
                throws TranslateException {
            int size = children.size();
            List<CompletableFuture<Output>> futures = new ArrayList<>(size);
            if (splitInputs) {
                List<Input> inputs = split(input);
                if (size == inputs.size()) {
                    for (int i = 0; i < size; ++i) {
                        futures.add(children.get(i).predict(ctx, inputs.get(i)));
                    }
                } else if (size == 1) {
                    for (int i = 0; i < size; ++i) {
                        futures.add(children.get(0).predict(ctx, inputs.get(i)));
                    }
                } else {
                    throw new TranslateException("split output size mismatch");
                }
            } else {
                for (Node node : children) {
                    futures.add(node.predict(ctx, input));
                }
            }

            Output output = new Output(200, "OK");
            try {
                if (mergeOutputs) {
                    NDList ndList = new NDList();
                    for (CompletableFuture<Output> f : futures) {
                        Output o = f.get();
                        ndList.addAll(o.getDataAsNDList(ctx.getNDManager()));
                    }
                    output.add("data", ndList);
                } else {
                    for (CompletableFuture<Output> f : futures) {
                        Output o = f.get();
                        for (Pair<String, BytesSupplier> pair : o.getContent()) {
                            output.add(pair.getKey(), pair.getValue());
                        }
                    }
                }
            } catch (InterruptedException | ExecutionException e) {
                throw new TranslateException(e);
            }

            CompletableFuture<Output> ret = new CompletableFuture<>();
            ret.complete(output);
            return ret;
        }
    }

    /** A {@code Node} reference to a model registered in the model server. */
    final class ModelRef extends Node {

        private String name;
        private String version;
        private transient InputProcessor inputProcessor;
        private transient OutputProcessor outputProcessor;
        private String inputProcessorClass;
        private String outputProcessorClass;

        /** {@inheritDoc} */
        @Override
        public CompletableFuture<Output> predict(TranslatorContext ctx, Input input)
                throws TranslateException {
            ctx.setAttachment("input", input);
            ModelManager modelManager = ModelManager.getInstance();
            initProcessor();
            if (inputProcessor != null) {
                input = inputProcessor.processInput(ctx, input);
            }
            ModelInfo modelInfo = modelManager.getModel(name, version, true);
            CompletableFuture<Output> future = new CompletableFuture<>();
            modelManager
                    .runJob(new Job(modelInfo, input))
                    .whenComplete(
                            (o, t) -> {
                                if (o != null) {
                                    if (outputProcessor != null) {
                                        try {
                                            future.complete(outputProcessor.processOutput(ctx, o));
                                        } catch (TranslateException e) {
                                            future.completeExceptionally(e);
                                        }
                                    } else {
                                        future.complete(o);
                                    }
                                }
                            })
                    .exceptionally(
                            t -> {
                                future.completeExceptionally(t);
                                return null;
                            });
            return future;
        }

        private void initProcessor() throws TranslateException {
            try {
                if (inputProcessorClass != null && inputProcessor == null) {
                    Class<?> clazz = Class.forName(inputProcessorClass);
                    Class<? extends InputProcessor> subclass =
                            clazz.asSubclass(InputProcessor.class);
                    Constructor<? extends InputProcessor> constructor = subclass.getConstructor();
                    inputProcessor = constructor.newInstance();
                }
                if (outputProcessorClass != null && outputProcessor == null) {
                    Class<?> clazz = Class.forName(outputProcessorClass);
                    Class<? extends OutputProcessor> subclass =
                            clazz.asSubclass(OutputProcessor.class);
                    Constructor<? extends OutputProcessor> constructor = subclass.getConstructor();
                    outputProcessor = constructor.newInstance();
                }
            } catch (ReflectiveOperationException e) {
                throw new TranslateException(e);
            }
        }
    }
}
