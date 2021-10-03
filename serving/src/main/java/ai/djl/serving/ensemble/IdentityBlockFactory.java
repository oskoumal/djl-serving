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

import ai.djl.Model;
import ai.djl.nn.Block;
import ai.djl.nn.BlockFactory;
import ai.djl.nn.Blocks;
import java.nio.file.Path;
import java.util.Map;

/** A {@link BlockFactory} class that creates an identity block. */
public class IdentityBlockFactory implements BlockFactory {

    private static final long serialVersionUID = 1L;

    /** {@inheritDoc} */
    @Override
    public Block newBlock(Model model, Path modelPath, Map<String, ?> arguments) {
        return Blocks.identityBlock();
    }
}
