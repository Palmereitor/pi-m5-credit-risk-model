| Variable | Interpretación probable |
|---|---|
| `tipo_credito` | Tipo de crédito solicitado. Los valores numéricos (4, 7, 9, etc.) son códigos categóricos, un número es un tipo de crédito, para el caso no es importante qué tipo de crédito. |
| `fecha_prestamo` | Fecha y hora en que se registró/otorgó el préstamo. |
| `capital_prestado` | Monto de capital originalmente prestado al cliente. |
| `plazo_meses` | Plazo acordado para cancelar el préstamo, expresado en meses. |
| `edad_cliente` | Edad del cliente al momento del préstamo. |
| `tipo_laboral` | Situación laboral del cliente, por ejemplo `Empleado` o `Independiente`. |
| `salario_cliente` | Ingreso/salario declarado por el cliente. |
| `total_otros_prestamos` | Monto total de otros préstamos/deudas que posee el cliente. |
| `cuota_pactada` | Valor de la cuota periódica acordada para el préstamo. |
| `puntaje` | Score o índice utilizado para evaluar al cliente. |
| `puntaje_datacredito` | Puntaje/score crediticio proveniente de DataCrédito. Otra fuente más de puntaje. |
| `cant_creditosvigentes` | Cantidad de créditos u obligaciones que el cliente tiene vigentes. |
| `huella_consulta` | Cantidad de consultas realizadas sobre el historial crediticio del cliente. Una consulta genera una "huella" en el historial. Se corre la consulta cuando el cliente va a distintas entidadaes a solicitar crédito y se busca su puntaje. |
| `saldo_mora` | Monto de obligaciones que se encuentra actualmente vencido/en mora. |
| `saldo_total` | Saldo total de las obligaciones/deudas reportadas. |
| `saldo_principal` | Saldo pendiente correspondiente al capital principal de las obligaciones. Lo que debe, vencido o no. |
| `saldo_mora_codeudor` | Monto en mora asociado a obligaciones en las que el cliente figura como codeudor. |
| `creditos_sectorFinanciero` | Cantidad de créditos/obligaciones registrados en el sector financiero. No es el monto en moneda, es la cantidad absoluta de créditos. El tipo de dato debe ser un entero. |
| `creditos_sectorCooperativo` | Cantidad de créditos/obligaciones registrados en el sector cooperativo. |
| `creditos_sectorReal` | Cantidad de créditos/obligaciones registrados en el sector real/comercial. Prendas sobre vehículos o hipotecas principalmente. |
| `promedio_ingresos_datacredito` | Promedio de ingresos reportados o estimados por DataCrédito. |
| `tendencia_ingresos` | Tendencia observada en los ingresos del cliente: `Creciente`, `Estable` o `Decreciente`. Se mide con el tiempo. Si durante algunos meses se nota que el cliente gana capacidad adquisitiva es porque sus ingresos mejorar y se lo categoriza como `creciente` o lo opuesto, como `decreciente`. |
| `Pago_atiempo` | Variable objetivo binaria. `1` = pago a tiempo y `0` = no pago a tiempo. |