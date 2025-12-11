<Train>
Input -> [Neural Network Layers] -> [Final Linear Layer] --(outputs)--> LOGITS
                                                                |
                                                                v
                                                       [Activation Function] --> FINAL INTERPRETABLE OUTPUT
                                                            (e.g., Softmax/Sigmoid)

Image encoder -> loss update(img_feat) -> point encoder
                                       -> sdmap encoder
                                       -> bev_feat = bev encoder(pt_feat, sdmap_feat) -> loss update(bev_feat) -> preds = head(bev_feat) -> head.loss(preds, gt) -> loss update(loss)


<Test>
Image encoder -> point encoder
              -> sdmap encoder
              -> bev_feat = bev encoder(pt_feat, sdmap_feat) - preds = head(bev_feat) -> head.decode(preds) : nms, etc