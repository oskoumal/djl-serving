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
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.ndarray.BytesSupplier;
import ai.djl.translate.TranslateException;
import ai.djl.translate.TranslatorContext;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.List;

public class SubImage implements OutputProcessor {

    @Override
    public Output processOutput(TranslatorContext ctx, Output output) throws TranslateException {
        try {
            Input input = (Input) ctx.getAttachment("input");
            byte[] buf = input.getData().getAsBytes();
            ImageFactory factory = ImageFactory.getInstance();
            Image img = factory.fromInputStream(new ByteArrayInputStream(buf));
            int width = img.getWidth();
            int height = img.getHeight();
            BytesSupplier data = output.getData();
            DetectedObjects detections = (DetectedObjects) data;
            List<DetectedObjects.DetectedObject> items = detections.items();
            Output ret = new Output(200, "OK");
            for (DetectedObjects.DetectedObject detection : items) {
                if ("person".equals(detection.getClassName())) {
                    BoundingBox box = detection.getBoundingBox();
                    Rectangle rect = box.getBounds();
                    int x = (int) (rect.getX() * width);
                    int y = (int) (rect.getY() * height);
                    int w = (int) (rect.getWidth() * width);
                    int h = (int) (rect.getHeight() * height);

                    Image subImage = img.getSubimage(x, y, w, h);
                    ByteArrayOutputStream bos = new ByteArrayOutputStream();
                    subImage.save(bos, "png");
                    bos.close();
                    ret.add(bos.toByteArray());
                }
            }
            return ret;
        } catch (IOException e) {
            throw new TranslateException(e);
        }
    }
}
