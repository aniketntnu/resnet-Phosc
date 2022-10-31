            if avgLoss.data.cpu().numpy()[0]:

                w=net.phoc[0].weight
                b=net.phoc[0].bias

                s1=sum(w[0]).cpu().detach().numpy()
                b1=sum(b).cpu().detach().numpy()

                ws=net.phos[0].weight
                bs=net.phos[0].bias

                ss1=sum(ws[0]).cpu().detach().numpy()
                bs1=sum(bs).cpu().detach().numpy()


