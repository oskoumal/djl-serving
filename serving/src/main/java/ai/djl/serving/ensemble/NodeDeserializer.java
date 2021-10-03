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

import com.google.gson.JsonDeserializationContext;
import com.google.gson.JsonDeserializer;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import java.lang.reflect.Type;

class NodeDeserializer implements JsonDeserializer<Node> {

    /** {@inheritDoc} */
    @Override
    public Node deserialize(JsonElement element, Type type, JsonDeserializationContext ctx) {
        JsonObject jsonObj = element.getAsJsonObject();
        JsonElement typeField = jsonObj.get("type");
        String nodeType = typeField == null ? null : typeField.getAsString();
        if ("sequence".equals(nodeType)) {
            return ctx.deserialize(element, Node.Sequence.class);
        } else if ("parallel".equals(nodeType)) {
            return ctx.deserialize(element, Node.Parallel.class);
        }
        return ctx.deserialize(element, Node.ModelRef.class);
    }
}
